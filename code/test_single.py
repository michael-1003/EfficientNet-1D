import os
import time
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipeline import NLUDataset
from src.configurator import read_config, select_data, select_model, select_loss
from src.utils import save_train_log, save_ckpt, load_ckpt
from src.evaluate import evaluate




#%%
parser = argparse.ArgumentParser()
parser.add_argument('--config_fname', type=str, default='sample')
parser.add_argument('--case_num', type=int, default=0)
args = parser.parse_args()


#%%
CONFIG_FNAME = args.config_fname
try:
    experiment_configs = read_config(CONFIG_FNAME)
except ValueError as e:
    print('There is no such configure file!')

RESULT_ROOT_DIR = '../results/' + CONFIG_FNAME

if not os.path.exists(RESULT_ROOT_DIR):
    raise(ValueError('There is no such expriments have configure file name!'))


#%%
def main(config):
    CASE_NUM        = config['case_num']

    DATASET         = config['dataset']
    NORMALIZATION   = config['normalization']

    BATCH_SIZE      = config['batch_size']
    MAX_EPOCH       = config['max_epoch']
    OPTIM_TYPE      = config['optimzer']
    LR              = config['learning_rate']
    LR_STEP         = config['lr_step']
    LR_DECAY        = config['lr_decay']
    L2_DECAY        = config['l2_decay']
    TB_STATE        = config['use_tensorboard']

    MODEL_NAME      = config['model_name']
    ALPHA           = config['alpha']
    BETA            = config['beta']
    GAMMA           = config['gamma']
    PHI             = config['phi']
    LOSS_FN         = config['loss_fn']
    KERNEL_SIZE     = config['kernel_size']

    result_dir = RESULT_ROOT_DIR + '/' + CASE_NUM
    ckpt_path = result_dir + '/' + 'checkpoint.pt'


    #%%
    data_fname, data_dim = select_data(DATASET)
    data_path = '../data/' + data_fname

    data_test = NLUDataset(data_path, mode='test', random_seed=42)
    dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    classes = data_test.labels
    num_classes =  len(classes)


    #%%
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if device=='cuda': print('Using GPU, %s' % torch.cuda.get_device_name(0))

    net = select_model(MODEL_NAME, data_dim, KERNEL_SIZE, num_classes, ALPHA, BETA, PHI)
    net.to(device)
    loss_fn = select_loss(LOSS_FN)


    #%%
    net, best_validation_acc = load_ckpt(ckpt_path, net)

    start_time = time.time()
    test_acc, test_loss = evaluate(dataloader_test, device, net, loss_fn)
    curr_time = time.time()
    ttt = curr_time - start_time
    tt1 = ttt / data_test.__len__()

    print('########################################################')
    print('# Test accuracy of %d: %.4f' % (CASE_NUM, test_acc))
    print("# Average %.6f s to process one input" % (tt1))
    print('########################################################')



#%%
if __name__ == "__main__":
    config = experiment_configs[args.case_num]
    main(config)

