'''
==============================================================

Train ket points net


==============================================================

Author:
version: v 0.70
Date:
note: 前 k 轮冻结训练 cov 的层      __C.TRAIN.LEARNING_RATE = 0.002
cmd: CUDA_VISIBLE_DEVICES=6,7 python3 train_kp3dgs_shapenet55.py
     CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_kp3dgs_shapenet55.py

2 gpus:  __C.CONST.DEVICE   = '6,7'
         model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
==============================================================
'''
from model_kp_3dgs import KP_3DGS,kp_3dgs_loss,means_cd_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import numpy as np
import json
import time
import utils.data_loaders
from pprint import pprint
from easydict import EasyDict as edict

from utils.metrics import Metrics
import utils.data_loaders
import utils.helpers
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from models.utils import fps_subsample

TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ----------------------------------------------------------------------------------------------------------------------
#
#           Arguments 
#       \******************/
#

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing SeedFormer', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--arch_model', type=str, default='seedformer_dim128', help='Model to use.')
parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=False, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
parser.add_argument('--mode', type=str, default='median', help='Testing mode [easy, median, hard].')
args = parser.parse_args()


def ShapeNet55Config():

    #######################
    # Configuration for PCN
    #######################

    __C                                              = edict()
    cfg                                              = __C

    #
    # Dataset Config
    #
    __C.DATASETS                                     = edict()
    __C.DATASETS.SHAPENET55                          = edict()
    __C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH       = '/home/ps/wcw_1999/codes/seedformer-master/codes/datasets/ShapeNet55-34/ShapeNet-55'
    __C.DATASETS.SHAPENET55.N_POINTS                 = 2048
    __C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH     = '/home/ps/wcw_1999/datasets/ShapeNet55/shapenet_pc/%s'

    #
    # Dataset
    #
    __C.DATASET                                      = edict()
    # Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT
    __C.DATASET.TRAIN_DATASET                        = 'ShapeNet55'
    __C.DATASET.TEST_DATASET                         = 'ShapeNet55'

    #
    # Constants
    #
    __C.CONST                                        = edict()

    __C.CONST.NUM_WORKERS                            = 8
    __C.CONST.N_INPUT_POINTS                         = 2048

    #
    # Directories
    #

    __C.DIR                                          = edict()
    __C.DIR.OUT_PATH                                 = '/home/ps/wcw_1999/codes/seedformer-master/results_kp3dgs'
    __C.DIR.TEST_PATH                                = '/home/ps/wcw_1999/codes/seedformer-master/test_kp3dgs'
    # __C.CONST.DEVICE                                 = '0, 1'
    __C.CONST.DEVICE                                 = ' 4,5,6,7'   # cmd: CUDA_VISIBLE_DEVICES=6,7 python3 train_kp3dgs_shapenet55.py 
    # __C.CONST.WEIGHTS                                = None # 'ckpt-best.pth'  # specify a path to run test and inference

    #
    # Network
    #
    __C.NETWORK                                      = edict()
    __C.NETWORK.UPSAMPLE_FACTORS                     = [1, 4, 4]

    #
    # Train
    #
    __C.TRAIN                                        = edict()
    __C.TRAIN.BATCH_SIZE                             = 200 # 320~480
    __C.TRAIN.N_EPOCHS                               = 400
    __C.TRAIN.LEARNING_RATE                          = 0.002
    __C.TRAIN.LR_DECAY                               = 100
    __C.TRAIN.WARMUP_EPOCHS                          = 40
    __C.TRAIN.GAMMA                                  = .5
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0

    #
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'


    return cfg


class Manager_kp:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, cfg):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # training dataset
        self.dataset = cfg.DATASET.TRAIN_DATASET

        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,     
                                              after_scheduler=self.scheduler_steplr)    # 前40轮学习率增长
        
        

        
        

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()

    def unpack_data(self, data):

        if self.dataset == 'ShapeNet':
            partial = data['partial_cloud']
            gt = data['gtcloud']
        elif self.dataset == 'ShapeNet55':
            # generate partial data online
            gt = data['gtcloud']
            _, npoints, _ = gt.shape
            partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
        else:
            raise ValueError('No method implemented for this dataset: {:s}'.format(self.dataset))

        return partial, gt
    
    def train(self, model, train_data_loader, val_data_loader, cfg):
        
        # 
        change_epoch = 5
        
        init_epoch = 0
        steps = 0
        
        # freeze_cov_layers
        # --------------------------------------------------------------------------------
        print("Freezing cov-related layers for the first 50 epochs...")
        model.module.freeze_cov_layers()  # 假设 model 有 freeze_cov_layers 方法
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)
        for name, param in model.named_parameters():
                    print(f"{name}: requires_grad = {param.requires_grad}")
        # LR
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)    # 前40轮学习率增长
        # --------------------------------------------------------------------------------
        


        # training record file
        print('Training Record:')
        self.train_record('n_itr, total_loss')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | #best_epoch best_metrics')
        torch.autograd.set_detect_anomaly(True)
        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx
            
            # unfreeze_cov_layers
            # ----------------------------------------------------------------------------
            if epoch_idx == change_epoch:
                print(f"[Epoch {epoch_idx}] Unfreezing cov-related layers...")
                model.module.unfreeze_cov_layers()  # 假设 model 有 unfreeze_cov_layers 方法
                model.module.freeze_means_layers()
                # Reinitialize optimizer to include unfrozen parameters
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                   lr=cfg.TRAIN.LEARNING_RATE,
                                                   weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                                   betas=cfg.TRAIN.BETAS)
                # 验证 freeze
                for name, param in model.named_parameters():
                    print(f"{name}: requires_grad = {param.requires_grad}")
                
                self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
                self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,     
                                              after_scheduler=self.scheduler_steplr)    # 前40轮学习率增长
            # ----------------------------------------------------------------------------

            # timer
            epoch_start_time = time.time()

            model.train()

            # Update learning rate

            self.lr_scheduler.step()
            
            
            

            # total cds
            total_t = 0
            
            ii = 0
            kk = 0
            base_path = '../test_kp/test_kp3dgs_cloud'

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)    # 获取 partial 点云    partial shape: torch.Size([B, 2048, 3])  gt shape:  torch.Size([224, 8192, 3])
                partial = partial.permute(0,2,1) # (B, 3, 2048)
                
                
                gt_downsample = fps_subsample(gt,1024).permute(0,2,1)  # (B, 3, 1024)
                means, sample_points = model(partial)
                # print("kp size: ", kp.shape)    # kp size: torch.Size([120, 24, 3])
                # exit()
                
                
                # if epoch_idx == 5:
                # kp_cut = np.squeeze(kp_3dgs)   # 去掉一个维度
                # tensor_cpu = kp_cut.cpu()      # 转换为 cpu 张量
                # kp_cpu = tensor_cpu.detach().numpy()    # 张量转 numpy
                # file_name = f'partial_{ii}.npy'
                # ii = ii + 1
                # file_path = os.path.join(base_path, file_name)
                # np.save(file_path, kp_cpu)     # 保存所有
                
                
                # gt_cut = np.squeeze(gt_downsample.permute(0,2,1))   # 去掉一个维度
                # gt_cut_cpu = gt_cut.cpu()      # 转换为 cpu 张量
                # gt_cpu = gt_cut_cpu.detach().numpy()    # 张量转 numpy
                # file_name2 = f'gt_{kk}.npy'
                # kk = kk + 1
                # file_path2 = os.path.join(base_path, file_name2)
                # np.save(file_path2, gt_cpu)     # 保存所有
                
                # if ii == 100:
                #     exit()
                
                
                
                if epoch_idx < change_epoch:
                    loss_total = means_cd_loss()(means, gt)
                else:
                    loss_total, mean_loss, kp_loss  = kp_3dgs_loss()(means, sample_points, gt)

                # loss_total, mean_loss, kp_loss  = kp_3dgs_loss()(means, sample_points, gt)


                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()
                
                total_t += loss_total.item()
                
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                if epoch_idx < change_epoch:
                    message = '{:d} loss_total: {:.4f} '.format(n_itr, loss_total.item())
                    self.train_record(message, show_info=True)
                else:
                    message = '{:d} loss_total:  {:.4f} mean_loss: {:.4f} kp_loss: {:.4f} '.format(n_itr, loss_total.item(), mean_loss.item(), kp_loss.item(), )
                    self.train_record(message, show_info=True)
                
                # message = '{:d} loss_total: {:.4f} kp_loss: {:.4f} mean_loss: {:.4f}'.format(n_itr, loss_total.item(), kp_loss.item(), mean_loss.item())
                # self.train_record(message, show_info=True)
                
                

            # avg cds
            avg_loss = total_t / n_batches

            epoch_end_time = time.time()

            # Training record
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, avg_loss))

            # Validate the current model
            #cd_eval = self.validate(cfg, model=model, val_data_loader=val_data_loader)
            #self.train_record('Testing scores = {:.4f}'.format(cd_eval))

            # Save checkpoints
            cd_eval = avg_loss
            if cd_eval < self.best_metrics:
                self.best_epoch = epoch_idx
                file_name = 'ckpt-best.pth' if cd_eval < self.best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': model.state_dict()
                }, output_path)

                print('Saved checkpoint to %s ...' % output_path)
                if cd_eval < self.best_metrics:
                    self.best_metrics = cd_eval

        # training end
        self.train_record_file.close()
        self.test_record_file.close()




def train_kp(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)  # cfg.DATASET.TEST_DATASET =  'ShapeNet55'

    # # test
    # train_data_set = train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN)
    # taxonomy_id, model_id, data = train_data_set[0]
    # print(len(train_data_set))             # 41952
    # print(f"taxonomy_id: {taxonomy_id}")   # taxonomy_id: 02828884       train.txt 中第一个
    # print(f"model_id: {model_id}")         # model_id: 3d2ee152db78b312e5a8eba5f6050bab
    # print(data)
    # for key, value in data.items():
    #     print(f"Shape of {key}: {value.shape}") # Shape of gtcloud: torch.Size([8192, 3])
    # quit()
    
    
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),  # datast 中有 41952 个元素，每个元素中：taxonomy_id, model_id, data，data 中 key: gtcloud value: [8192, 3]
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,   # 指定每个 batch 中的样本数量 cfg: 48
                                                    num_workers=cfg.CONST.NUM_WORKERS, # load data 的进程数量 cfg: 8
                                                    collate_fn=utils.data_loaders.collate_fn, # 定义了如何将多个数据样本组合成一个批次
                                                    pin_memory=True,  # 性能优化
                                                    shuffle=True,     # 每个 epoch 开始时对数据进行打乱
                                                    drop_last=False)  # 最后一个批次不完整，也会被包含在训练中
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(     # 验证集，用的 test dataset
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
    cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, TRAIN_NAME+timestr)
    cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints')
    cfg.DIR.LOGS = cfg.DIR.OUT_PATH
    print('Saving outdir: {}'.format(cfg.DIR.OUT_PATH))
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # save config file
    pprint(cfg)
    config_filename = os.path.join(cfg.DIR.LOGS, 'config.json')
    with open(config_filename, 'w') as file:
        json.dump(cfg, file, indent=4, sort_keys=True)

    # Save Arguments
    torch.save(args, os.path.join(cfg.DIR.LOGS, 'args_training.pth'))

    #######################
    # Prepare Network Model
    #######################

    model = KP_3DGS(k = 64)
    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()  # 设置多 GPU？
    
    ####################### training
    manager = Manager_kp(model, cfg)

    # Start training
    manager.train(model, train_data_loader, val_data_loader, cfg)
    

def test_kp(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)  # cfg.DATASET.TEST_DATASET =  'ShapeNet55'

    # # test
    # train_data_set = train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN)
    # taxonomy_id, model_id, data = train_data_set[0]
    # print(len(train_data_set))             # 41952
    # print(f"taxonomy_id: {taxonomy_id}")   # taxonomy_id: 02828884       train.txt 中第一个
    # print(f"model_id: {model_id}")         # model_id: 3d2ee152db78b312e5a8eba5f6050bab
    # print(data)
    # for key, value in data.items():
    #     print(f"Shape of {key}: {value.shape}") # Shape of gtcloud: torch.Size([8192, 3])
    # quit()
    
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),  # datast 中有 41952 个元素，每个元素中：taxonomy_id, model_id, data，data 中 key: gtcloud value: [8192, 3]
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,   # 指定每个 batch 中的样本数量 cfg: 48
                                                    num_workers=cfg.CONST.NUM_WORKERS, # load data 的进程数量 cfg: 8
                                                    collate_fn=utils.data_loaders.collate_fn, # 定义了如何将多个数据样本组合成一个批次
                                                    pin_memory=True,  # 性能优化
                                                    shuffle=True,     # 每个 epoch 开始时对数据进行打乱
                                                    drop_last=False)  # 最后一个批次不完整，也会被包含在训练中
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(     # 验证集，用的 test dataset
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
    cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, TRAIN_NAME+timestr)
    cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints')
    cfg.DIR.LOGS = cfg.DIR.OUT_PATH
    print('Saving outdir: {}'.format(cfg.DIR.OUT_PATH))
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # save config file
    pprint(cfg)
    config_filename = os.path.join(cfg.DIR.LOGS, 'config.json')
    with open(config_filename, 'w') as file:
        json.dump(cfg, file, indent=4, sort_keys=True)

    # Save Arguments
    torch.save(args, os.path.join(cfg.DIR.LOGS, 'args_training.pth'))

    #######################
    # Prepare Network Model
    #######################

    model = KPN(24) 
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])   # 设置多 GPU？
    
    ####################### training
    manager = Manager_kp(model, cfg)

    # Start training
    manager.train(model, train_data_loader, val_data_loader, cfg)


if __name__ == '__main__':
    # Check python version
    #seed = 1
    #set_seed(seed)
    
    print('cuda available ', torch.cuda.is_available())

    # Init config
    cfg = ShapeNet55Config()

    # setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    
    train_kp(cfg)