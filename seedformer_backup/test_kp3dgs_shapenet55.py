'''
==============================================================

Train ket points net


==============================================================

Author:
Date:
note: flii checkpoint = torch.load
           cfg.DIR.PRETRAIN
           
cmd: CUDA_VISIBLE_DEVICES=6,7 python3 test_kp3dgs_shapenet55.py
==============================================================
'''
from model_kp_3dgs import KP_3DGS,kp_3dgs_loss
from key_point_net import KPN,ppro_cd_loss
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
from utils.loss_utils import get_loss
from utils.metrics import Metrics
import utils.data_loaders
import utils.helpers
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from models.utils import fps_subsample
from utils.average_meter import AverageMeter
from importlib import import_module
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ----------------------------------------------------------------------------------------------------------------------
#
#           Arguments 
#       \******************/
#

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing SeedFormer', help='description')
parser.add_argument('--net_model', type=str, default='model_kp_3dgs', help='Import module.')
parser.add_argument('--arch_model', type=str, default='kp_24', help='Model to use.')
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
    __C.DIR.TEST_PATH                                = '/home/ps/wcw_1999/codes/seedformer-master/test_kp'
    __C.CONST.DEVICE                                 = ' 6, 7'
    # __C.CONST.DEVICE                                 = '0, 1, 2, 3, 4, 5, 6, 7'
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
    __C.TRAIN.BATCH_SIZE                             = 160
    __C.TRAIN.N_EPOCHS                               = 400
    __C.TRAIN.LEARNING_RATE                          = 0.001
    __C.TRAIN.LR_DECAY                               = 100
    __C.TRAIN.WARMUP_EPOCHS                          = 20
    __C.TRAIN.GAMMA                                  = .5
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0

    #
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'


    return cfg



def chamfer_sqrt(p1, p2):
    chamfer_dist = chamfer_3DDist()
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

def kp_gt_loss(kp, gt):
    CD = chamfer_sqrt
    B1,N1,_ = kp.shape
    # gt1 = fps_subsample(gt,N1)
    cd = CD(kp, gt)

    return cd*1e3
    

    
    
    
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
                                              after_scheduler=self.scheduler_steplr)

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
    
        
    def test_shapenet55(self, cfg, model=None, test_data_loader=None, outdir=None, mode=None):
        """
        Testing Method for dataset shapenet-55/34
        """

        from models.utils import fps_subsample
        
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Eval settings
        crop_ratio = {
            'easy': 1/4,
            'median' :1/2,
            'hard':3/4
        }
        choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                  torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]

        # Switch models to evaluation mode
        model.eval()   # 将模型设置为评估模式

        n_samples = len(test_data_loader)
        # print(n_samples)

        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        print('Start evaluating (mode: {:s}) ...'.format(mode))
        ii = 0
        kk = 0
        pp = 0
        ff = 0
        base_path = '../test_kp/test_kp3dgs_cloud'
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]
            
           


            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # generate partial data online
                gt = data['gtcloud']

                
                    
                _, npoints, _ = gt.shape
                
                # partial clouds from fixed viewpoints
                num_crop = int(npoints * crop_ratio[mode])  # 设置残缺点云点的数量
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # print("partial shape: ",partial.shape)
                    partial = fps_subsample(partial, 2048).permute(0,2,1)  # 对加载的数据进行下采样当残缺点云  4096

                    # print(gt.shape)

                    
                    means, kp_3dgs, sample_points_re = model(partial)
                    B, N, sample_num,_ = kp_3dgs.shape
                    kp_3dgs = kp_3dgs.reshape(B, N * sample_num, 3)

                    # loss = kp_gt_loss(kp_3dgs, gt)
                    # # print(loss)
                    # # get loss
                    # cd_gt = loss * 1e3
                    # test_losses.update([cd_gt])

                    
                    # get all metrics
                    _metrics = Metrics.get(sample_points_re, gt)   # 获得 test 输出的 ChamferDistance ChamferDistanceL1 F-Score
                    test_metrics.update(_metrics)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)




        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        # message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


            # return test_losses.avg(3)
        



def test_kp(cfg):  # 测试
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)   # cfg.DATASET.TEST_DATASET = 'ShapeNet55'

    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=1,    # 每个数据批次的大小为1
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)


    # # Path for pretrained model
    # if args.pretrained == '':
    #     list_trains = os.listdir(cfg.DIR.OUT_PATH)
    #     # print(list_trains)   ['train_shapenet55_Log_2024_09_07_01_16_07']  
    #     list_pretrained = [train_name for train_name in list_trains if train_name.startswith(TRAIN_NAME+'_Log')]
    #     if len(list_pretrained) != 1:
    #         raise ValueError('Find {:d} models. Please specify a path for testing.'.format(len(list_pretrained)))

    #     cfg.DIR.PRETRAIN = list_pretrained[0]
    #     print(cfg.DIR.PRETRAIN)
    #     exit()
    # else:
    #     cfg.DIR.PRETRAIN = args.pretrained

    cfg.DIR.PRETRAIN = 'train_kp3dgs_shapenet55_Log_2024_12_06_01_53_06'
    
    # Set up folders for logs and checkpoints
    testset_name = cfg.DATASETS.SHAPENET55.CATEGORY_FILE_PATH    # 测试集文件名列表 test.txt
    testset_name = os.path.basename(testset_name.strip('/'))
    cfg.DIR.TEST_PATH = os.path.join(cfg.DIR.TEST_PATH, cfg.DIR.PRETRAIN, testset_name, args.mode)
    # print(cfg.DIR.TEST_PATH)     /home/ps/wcw_1999/codes/seedformer-master/test_kp/train_kp_shapenet55_Log_2024_10_10_14_38_39/ShapeNet-55/median
    # exit()
    cfg.DIR.RESULTS = os.path.join(cfg.DIR.TEST_PATH, 'outputs')
    cfg.DIR.LOGS = cfg.DIR.TEST_PATH
    print('Saving outdir: {}'.format(cfg.DIR.TEST_PATH))
    if not os.path.exists(cfg.DIR.RESULTS):
        os.makedirs(cfg.DIR.RESULTS)

 

    #######################
    # Prepare Network Model
    #######################


    model = KP_3DGS(k = 64)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()  # 设置多 GPU？

    # load pretrained model
    # cfg.CONST.WEIGHTS = os.path.join(cfg.DIR.OUT_PATH, cfg.DIR.PRETRAIN, 'checkpoints', 'ckpt-best.pth')
    # print(cfg.CONST.WEIGHTS)
    # print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load('../results_kp3dgs/train_kp3dgs_shapenet55_Log_2024_12_06_01_53_06/checkpoints/ckpt-best.pth')   # load check points
    model.load_state_dict(checkpoint['model'])

    ##################
    # Training Manager
    ##################

    manager = Manager_kp(model, cfg)

    # Start training
    manager.test_shapenet55(cfg, model, val_data_loader, outdir=cfg.DIR.RESULTS if args.output else None, mode=args.mode)




if __name__ == '__main__':
    # Check python version
    #seed = 1
    #set_seed(seed)
    
    print('cuda available ', torch.cuda.is_available())

    # Init config
    cfg = ShapeNet55Config()

    # setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    
    test_kp(cfg)