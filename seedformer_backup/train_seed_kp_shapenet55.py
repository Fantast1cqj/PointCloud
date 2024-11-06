'''
==============================================================

SeedFormer + key poins 
-> Training on ShapeNet-55/34

==============================================================

Author:
Date:

==============================================================
'''


import argparse
import os
import numpy as np
import torch
import json
import time
import utils.data_loaders
from easydict import EasyDict as edict
from importlib import import_module
from pprint import pprint
# from manager import Manager
from manager_seed_kp import Manager


TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ----------------------------------------------------------------------------------------------------------------------
#
#           Arguments 
#       \******************/
#

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing SeedFormer', help='description')
parser.add_argument('--net_model', type=str, default='model_addkp', help='Import module.')
parser.add_argument('--kp_net_model', type=str, default='key_point_net', help='Import kp module.')
parser.add_argument('--arch_model', type=str, default='seedformer_dim128', help='Model to use.')
parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=False, help='Output testing results.')       # 是否输出点云
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
    __C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH       = './datasets/ShapeNet55-34/ShapeNet-55/'
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
    __C.DIR.OUT_PATH                                 = '../results'
    __C.DIR.TEST_PATH                                = '../test'
    # __C.CONST.DEVICE                                 = '0, 1'
    __C.CONST.DEVICE                                 = '0, 1, 2, 3, 4, 5'
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
    __C.TRAIN.BATCH_SIZE                             = 200     # 48
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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def train_net(cfg):
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

    Model = import_module(args.net_model)  # 在当前目录下找 model_addkp.py
    kp_Model = import_module(args.kp_net_model) # 在当前目录下找 key_point_net.py
    # print(Model)    # <module 'model_addkp' from '/home/ps/wcw_1999/codes/seedformer-master/codes/model_addkp.py'>
    # print(kp_Model) # <module 'key_point_net' from '/home/ps/wcw_1999/codes/seedformer-master/codes/key_point_net.py'>
    # quit()
    model = Model.__dict__[args.arch_model](up_factors=cfg.NETWORK.UPSAMPLE_FACTORS)   # args.arch_model = seedformer_dim128 seedformer_dim128 在 model_addkp.py 中
    kp_model = kp_Model.__dict__['kp_128']()
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        kp_model = torch.nn.DataParallel(kp_model).cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])   # 设置多 GPU？

    # load existing model
    if 'WEIGHTS' in cfg.CONST:
        print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        print('Recover complete. Current epoch = #%d; best metrics = %s.' % (checkpoint['epoch_index'], checkpoint['best_metrics']))

    kp_net_checkpoint = torch.load('../results_kp/train_kp_shapenet55_Log_2024_10_22_11_08_37/checkpoints/ckpt-best.pth')
    kp_model.load_state_dict(kp_net_checkpoint['model'])    # ???
    ##################
    # Training Manager
    ##################

    manager = Manager(model, cfg)

    # Start training
    manager.train(model, kp_model, train_data_loader, val_data_loader, cfg)   # 输入：model, train data, val data, cfg


def test_net(cfg):  # 测试
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


    # Path for pretrained model
    if args.pretrained == '':
        list_trains = os.listdir(cfg.DIR.OUT_PATH)
        # print(list_trains)   ['train_shapenet55_Log_2024_09_07_01_16_07']  
        list_pretrained = [train_name for train_name in list_trains if train_name.startswith(TRAIN_NAME+'_Log')]
        if len(list_pretrained) != 1:
            raise ValueError('Find {:d} models. Please specify a path for testing.'.format(len(list_pretrained)))

        cfg.DIR.PRETRAIN = list_pretrained[0]
    else:
        cfg.DIR.PRETRAIN = args.pretrained


    # Set up folders for logs and checkpoints
    testset_name = cfg.DATASETS.SHAPENET55.CATEGORY_FILE_PATH    # 测试集文件名列表 test.txt
    testset_name = os.path.basename(testset_name.strip('/'))
    cfg.DIR.TEST_PATH = os.path.join(cfg.DIR.TEST_PATH, cfg.DIR.PRETRAIN, testset_name, args.mode)
    # print(cfg.DIR.TEST_PATH)    ../test/train_shapenet55_Log_2024_09_07_01_16_07/ShapeNet-55/median
    cfg.DIR.RESULTS = os.path.join(cfg.DIR.TEST_PATH, 'outputs')
    cfg.DIR.LOGS = cfg.DIR.TEST_PATH
    print('Saving outdir: {}'.format(cfg.DIR.TEST_PATH))
    if not os.path.exists(cfg.DIR.RESULTS):
        os.makedirs(cfg.DIR.RESULTS)


    #######################
    # Prepare Network Model
    #######################
    

    Model = import_module(args.net_model)  #
    kp_Model = import_module(args.kp_net_model)
    
    model = Model.__dict__[args.arch_model](up_factors=cfg.NETWORK.UPSAMPLE_FACTORS)
    kp_model = kp_Model.__dict__['kp_128']()
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        kp_model = torch.nn.DataParallel(kp_model).cuda()

    # load pretrained model
    # cfg.CONST.WEIGHTS = os.path.join(cfg.DIR.OUT_PATH, cfg.DIR.PRETRAIN, 'checkpoints', 'ckpt-best.pth')
    # print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    
    kp_net_checkpoint = torch.load('../results_kp/train_kp_shapenet55_Log_2024_10_22_11_08_37/checkpoints/ckpt-best.pth')# load checkpoint
    kp_model.load_state_dict(kp_net_checkpoint['model'])
    
    checkpoint = torch.load('../results/train_seed_kp_shapenet55_Log_2024_10_25_08_55_25/checkpoints/ckpt-best.pth')     # load checkpoint
    model.load_state_dict(checkpoint['model'])

    ##################
    # Training Manager
    ##################

    manager = Manager(model, cfg)

    # Start training
    manager.test(cfg, model, kp_model, val_data_loader, outdir=cfg.DIR.RESULTS if args.output else None, mode=args.mode)
        

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    if not args.test and not args.inference:
        train_net(cfg)
    else:
        if args.test:
            test_net(cfg)
        else:
            inference_net(cfg)

