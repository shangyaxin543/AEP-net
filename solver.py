import os
import time
import numpy as np
from PIL import Image


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

#from prep import printProgressBar
from networks import RED_CNN
from models import DnCNN
from measure import compute_measure


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.result = args.result
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.input_path = args.input_path
        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        self.REDCNN = RED_CNN()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result,path):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray)
        ax[0].set_title('LR', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray)
        ax[2].set_title('HR', fontsize=30)

        f.savefig(os.path.join(self.result, 'fig', path))
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs):
            
            self.REDCNN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1
                
                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred,initial = self.REDCNN(x)
                #print('pred.shape',pred.shape)
                loss1 = self.criterion(pred, y)
                loss2 = self.criterion(initial, y)
                loss = 0.1*loss1+loss2
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                    
                    
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    train_losses.append(loss.item())
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))


    def test(self):
        del self.REDCNN
        # load
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        i=0
        psnr = []
        ssim = []
        rmse = []
        with torch.no_grad():
            for i, (x, y, path) in enumerate(self.data_loader):
                [_,shape_x,shape_y] = x.shape
                
                #name=x.name()
                #x = x.unsqueeze(0).float().to(self.device)
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                pred, initial_seg= self.REDCNN(x)
                x = x.view(shape_x,shape_y).cpu().detach()
                y = y.view(shape_x,shape_y).cpu().detach()
                pred = pred.view(shape_x,shape_y).cpu().detach()
                initial_seg = initial_seg.view(shape_x,shape_y).cpu().detach()
                
                # denormalize, truncate
                #x = self.trunc(x.view(shape_, shape_).cpu().detach())
                #y = self.trunc(y.view(shape_, shape_).cpu().detach())
                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]

                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]
                
                # psnr.append(pred_result[0])
                # np.savetxt('./metricx/psnr.txt',np.array(psnr),fmt = '%s')
                # ssim.append(pred_result[1])
                # np.savetxt('./metricx/ssim.txt',np.array(ssim),fmt = '%s')
                # rmse.append(pred_result[2])
                # np.savetxt('./metricx/rmse.txt',np.array(rmse),fmt = '%s')

                path = path[0].split("/")[-1]
                print(path)
                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result,path)

                print('\n')
                pred=pred.numpy()
                image=Image.fromarray(pred)
                if image.mode == "F":
                    image = image.convert('L')
                    
                image.save(os.path.join(self.result, 'test', path))
                initial_seg=initial_seg.numpy()*255
                initial_seg=Image.fromarray(initial_seg)
                if initial_seg.mode == "F":
                    initial_seg = initial_seg.convert('L')   
                initial_seg.save(os.path.join(self.result, 'initial_seg', path))
                print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader)))
                print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                  pred_ssim_avg/len(self.data_loader), 
                                                                                                  pred_rmse_avg/len(self.data_loader)))
               
        