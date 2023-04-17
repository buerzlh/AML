import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from model import model
from config.config import cfg, cfg_from_file, cfg_from_list
from prepare_data import *
import sys
import pprint

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights1', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--weights2', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train(args):
    bn_domain_map = {}
    dataloaders = prepare_data()
    num_domains_bn = 2

    # initialize model
    model_state_dict_1 = None
    model_state_dict_2 = None
    fx_pretrained = True
    resume_dict = None

    net_1 = model.danet(num_classes=cfg.DATASET.NUM_CLASSES, 
                 state_dict=model_state_dict_1,
                 feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR, 
                 frozen=[cfg.TRAIN.STOP_GRAD], 
                 fx_pretrained=fx_pretrained, 
                 dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                 fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS, 
                 num_domains_bn=num_domains_bn)

    net_2 = model.danet(num_classes=cfg.DATASET.NUM_CLASSES, 
                 state_dict=model_state_dict_2,
                 feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR, 
                 frozen=[cfg.TRAIN.STOP_GRAD], 
                 fx_pretrained=fx_pretrained, 
                 dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                 fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS, 
                 num_domains_bn=num_domains_bn)

    net_1 = torch.nn.DataParallel(net_1)
    net_2 = torch.nn.DataParallel(net_2)
    if torch.cuda.is_available():
       net_1.cuda()
       net_2.cuda()


    # initialize solver
    from solver.AML import AML as Solver
    train_solver = Solver(net_1, net_2, dataloaders, bn_domain_map=bn_domain_map, resume=resume_dict)

    # train 
    train_solver.solve()
    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    task = cfg.DATASET.SOURCE_NAME[0]+cfg.DATASET.TARGET_NAME[0]
    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.DATASET.NAME,task,'1',)
    if not os.path.exists(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)
    from logger import make_print_to_file
    make_print_to_file(cfg.SAVE_DIR, os.path.basename(__file__))
    print('Called with args:')
    print(args)


    print('Using config:')
    pprint.pprint(cfg)
    
    print('Output will be saved to %s.' % cfg.SAVE_DIR)
    train(args)
