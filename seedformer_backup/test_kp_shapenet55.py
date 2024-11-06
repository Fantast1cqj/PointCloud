'''
==============================================================

Train ket points net


==============================================================

Author:
Date:

==============================================================
'''
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

from utils.metrics import Metrics
import utils.data_loaders
import utils.helpers
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from models.utils import fps_subsample
from utils.average_meter import AverageMeter
from importlib import import_module

TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ----------------------------------------------------------------------------------------------------------------------
#
#           Arguments 
#       \******************/
#

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing SeedFormer', help='description')
parser.add_argument('--net_model', type=str, default='key_point_net', help='Import module.')
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
    __C.DIR.OUT_PATH                                 = '/home/ps/wcw_1999/codes/seedformer-master/results_kp'
    __C.DIR.TEST_PATH                                = '/home/ps/wcw_1999/codes/seedformer-master/test_kp'
    __C.CONST.DEVICE                                 = '0, 1'
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
    __C.TRAIN.BATCH_SIZE                             = 48
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
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        print('Start evaluating (mode: {:s}) ...'.format(mode))
        ii = 0
        kk = 0
        base_path = '../test_kp/test_kp_cloud'
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]
            
           


            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # generate partial data online
                gt = data['gtcloud']
                
                
                # print("HERE!!!!!!!!!")
                # print(gt.shape)   # ([1, 8192, 3])
                
                    
                _, npoints, _ = gt.shape
                
                # partial clouds from fixed viewpoints
                num_crop = int(npoints * crop_ratio[mode])  # 设置残缺点云点的数量
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    partial = fps_subsample(partial, 2048)  # 对加载的数据进行下采样当残缺点云

                    
                    # # print("partial!!!!!!!!!")
                    # # print(partial.shape)   # ([1, 2048, 3])
                    # partial_cut = np.squeeze(partial)   # 去掉一个维度
                    # tensor_cpu = partial_cut.cpu()      # 转换为 cpu 张量
                    # partial_cpu = tensor_cpu.numpy()    # 张量转 numpy
                    # # print(partial_cut.shape) # ([2048, 3])
                    # file_name = f'partial_{ii}.npy'
                    # ii = ii + 1
                    # file_path = os.path.join(base_path, file_name)
                    # np.save(file_path, partial_cpu)     # 保存所有
                    # # print(f'Array {ii} is saved to {file_path}')
                    
                
                    gt_downsample = fps_subsample(gt,1024).permute(0,2,1)
                    kp,pp = model(gt_downsample)
                    
            
                    # print("partial!!!!!!!!!")
                    # print(partial.shape)   # ([1, 2048, 3])
                    kp_cut = np.squeeze(kp)   # 去掉一个维度
                    tensor_cpu = kp_cut.cpu()      # 转换为 cpu 张量
                    kp_cpu = tensor_cpu.numpy()    # 张量转 numpy
                    file_name = f'partial_{ii}.npy'
                    ii = ii + 1
                    file_path = os.path.join(base_path, file_name)
                    np.save(file_path, kp_cpu)     # 保存所有
                    
                    
                    
                    
                    
                    gt_downsample = gt_downsample.permute(0,2,1)
                    gt_cut = np.squeeze(gt_downsample)   # 去掉一个维度
                    gt_cut_cpu = gt_cut.cpu()      # 转换为 cpu 张量
                    gt_cpu = gt_cut_cpu.numpy()    # 张量转 numpy
                    file_name2 = f'gt_{kk}.npy'
                    kk = kk + 1
                    file_path2 = os.path.join(base_path, file_name2)
                    np.save(file_path2, gt_cpu)     # 保存所有
                    # print(f'Array {ii} is saved to {file_path}')
                    
                    
                    # pcds_pred = model(partial.contiguous())
                    
                    
                    
                    # loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=False) # L2

                    # get loss
                    # cdc = losses[0].item() * 1e3
                    # cd1 = losses[1].item() * 1e3
                    # cd2 = losses[2].item() * 1e3
                    # cd3 = losses[3].item() * 1e3
                    # partial_matching = losses[4].item() * 1e3
                    # test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                    # get all metrics
            #         _metrics = Metrics.get(pcds_pred[-1], gt)   # 获得 test 输出的 ChamferDistance ChamferDistanceL1 F-Score
            #         test_metrics.update(_metrics)
            #         if taxonomy_id not in category_metrics:
            #             category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            #         category_metrics[taxonomy_id].update(_metrics)

            #         # output to file
            #         if outdir:
            #             if not os.path.exists(os.path.join(outdir, taxonomy_id)):
            #                 os.makedirs(os.path.join(outdir, taxonomy_id))
            #             if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
            #                 os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
            #             # save pred, gt, partial pcds 
            #             pred = pcds_pred[-1]
            #             for mm, model_name in enumerate(model_id):
            #                 output_file = os.path.join(outdir, taxonomy_id, model_name+'_{:02d}'.format(partial_id))
            #                 write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
            #                 write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
            #                 write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
            #                 # output img files
            #                 img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
            #                 output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
            #                 output_img = (output_img*255).astype('uint8')
            #                 im = Image.fromarray(output_img)
            #                 im.save(img_filename)


            # # Record category results
            # self.train_record('============================ TEST RESULTS ============================')
            # self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

            # for taxonomy_id in category_metrics:
            #     message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            #     message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            #     mclass_metrics.update(category_metrics[taxonomy_id].avg())
            #     self.train_record(message)

            # self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
            # self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

            # # record testing results
            # message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
            # self.test_record(message)


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

    cfg.DIR.PRETRAIN = 'train_kp_shapenet55_Log_2024_10_10_14_38_39'
    
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

    Model = import_module(args.net_model)  #
    model = Model.__dict__['kp_128']()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model
    # cfg.CONST.WEIGHTS = os.path.join(cfg.DIR.OUT_PATH, cfg.DIR.PRETRAIN, 'checkpoints', 'ckpt-best.pth')
    # print(cfg.CONST.WEIGHTS)
    # print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load('../results_kp/train_kp_shapenet55_Log_2024_10_22_11_08_37/checkpoints/ckpt-best.pth')   # load check points
    model.load_state_dict(checkpoint['model'])

    ##################
    # Training Manager
    ##################

    manager = Manager_kp(model, cfg)

    # Start training
    manager.test_shapenet55(cfg, model, val_data_loader, outdir=cfg.DIR.RESULTS if args.output else None, mode=args.mode)




# def test_kp(cfg):
#     # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
#     torch.backends.cudnn.benchmark = True

#     ########################
#     # Load Train/Val Dataset
#     ########################

#     train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
#     val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)  # cfg.DATASET.TEST_DATASET =  'ShapeNet55'

#     # # test
#     # train_data_set = train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN)
#     # taxonomy_id, model_id, data = train_data_set[0]
#     # print(len(train_data_set))             # 41952
#     # print(f"taxonomy_id: {taxonomy_id}")   # taxonomy_id: 02828884       train.txt 中第一个
#     # print(f"model_id: {model_id}")         # model_id: 3d2ee152db78b312e5a8eba5f6050bab
#     # print(data)
#     # for key, value in data.items():
#     #     print(f"Shape of {key}: {value.shape}") # Shape of gtcloud: torch.Size([8192, 3])
#     # quit()
    
#     train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
#         utils.data_loaders.DatasetSubset.TRAIN),  # datast 中有 41952 个元素，每个元素中：taxonomy_id, model_id, data，data 中 key: gtcloud value: [8192, 3]
#                                                     batch_size=cfg.TRAIN.BATCH_SIZE,   # 指定每个 batch 中的样本数量 cfg: 48
#                                                     num_workers=cfg.CONST.NUM_WORKERS, # load data 的进程数量 cfg: 8
#                                                     collate_fn=utils.data_loaders.collate_fn, # 定义了如何将多个数据样本组合成一个批次
#                                                     pin_memory=True,  # 性能优化
#                                                     shuffle=True,     # 每个 epoch 开始时对数据进行打乱
#                                                     drop_last=False)  # 最后一个批次不完整，也会被包含在训练中
#     val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(     # 验证集，用的 test dataset
#         utils.data_loaders.DatasetSubset.TEST),
#                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
#                                                   num_workers=cfg.CONST.NUM_WORKERS//2,
#                                                   collate_fn=utils.data_loaders.collate_fn,
#                                                   pin_memory=True,
#                                                   shuffle=False)

#     # Set up folders for logs and checkpoints
#     timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
#     cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, TRAIN_NAME+timestr)
#     cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints')
#     cfg.DIR.LOGS = cfg.DIR.OUT_PATH
#     print('Saving outdir: {}'.format(cfg.DIR.OUT_PATH))
#     if not os.path.exists(cfg.DIR.CHECKPOINTS):
#         os.makedirs(cfg.DIR.CHECKPOINTS)

#     # save config file
#     pprint(cfg)
#     config_filename = os.path.join(cfg.DIR.LOGS, 'config.json')
#     with open(config_filename, 'w') as file:
#         json.dump(cfg, file, indent=4, sort_keys=True)

#     # Save Arguments
#     torch.save(args, os.path.join(cfg.DIR.LOGS, 'args_training.pth'))

#     #######################
#     # Prepare Network Model
#     #######################

#     model = KPN(24) 
#     if torch.cuda.is_available():
#         model = torch.nn.DataParallel(model).cuda()
#         # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])   # 设置多 GPU？
    
#     ####################### training
#     manager = Manager_kp(model, cfg)

#     # Start training
#     manager.train(model, train_data_loader, val_data_loader, cfg)


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