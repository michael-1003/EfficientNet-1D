import os
import time
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from configurator import read_config, select_data, select_model, select_loss, select_optimizer

from src.pipeline import NLUDataset
from src.utils import make_dir, save_train_log, save_ckpt, load_ckpt
from src.train import train_1epoch
from src.evaluate import evaluate


#%% Input args
# Run this file in background: nohup python main.py --config_fname=sample > log_sample.txt &
parser = argparse.ArgumentParser()
parser.add_argument('--config_fname', type=str, default='sample')
parser.add_argument('--overwrite', type=bool, default=True)
args = parser.parse_args()


#%%
CONFIG_FNAME = args.config_fname
try:
    experiment_configs = read_config(CONFIG_FNAME)
except ValueError as e:
    print('There is no such configure file!')

RESULT_ROOT_DIR = make_dir('../results', CONFIG_FNAME, overwrite=args.overwrite)


#%%
def main(config):
    CASE_NUM        = config['case_num']

    DATASET         = config['dataset']
    NORMALIZATION   = config['normalization']

    BATCH_SIZE      = config['batch_size']
    MAX_EPOCH       = config['max_epoch']
    OPTIM_TYPE      = config['optimizer']
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

    result_dir = make_dir(RESULT_ROOT_DIR, str(CASE_NUM), overwrite=args.overwrite)
    ckpt_path = result_dir + '/' + 'checkpoint.pt'


    # =============================================== Select data and construct
    data_fname, data_dim = select_data(DATASET)
    data_path = '../data/' + data_fname

    data_train = NLUDataset(data_path, mode='train', normalization=NORMALIZATION, random_seed=42)
    dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    data_valid = NLUDataset(data_path, mode='valid', normalization=NORMALIZATION, random_seed=42)
    dataloader_valid = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    data_test = NLUDataset(data_path, mode='test', normalization=NORMALIZATION, random_seed=42)
    dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    num_train_samples = data_train.__len__()
    classes = data_train.labels
    num_classes =  len(classes)


    # =============================================== Initialize model and optimizer
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if device=='cuda': print('Using GPU, %s' % torch.cuda.get_device_name(0))

    net = select_model(MODEL_NAME, data_dim, KERNEL_SIZE, num_classes, ALPHA, BETA, PHI)
    net.to(device)
    loss_fn = select_loss(LOSS_FN)
    optimizer = select_optimizer(OPTIM_TYPE, net.parameters(), LR, L2_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=LR_STEP, gamma=LR_DECAY)


    # =============================================== Train
    it = 0
    train_losses, valid_losses, valid_accs = {}, {}, {}
    best_validation_acc = 0
    log_term = 5

    for epoch in range(MAX_EPOCH):
        #------------------------------------------------ One epoch start
        one_epoch_start = time.time()
        print('Epoch {} / Learning Rate: {:.0e}'.format(epoch,scheduler.get_lr()[0]))        
        #------------------------------------------------ Train
        train_losses, it, net, optimizer, scheduler \
            = train_1epoch(dataloader_train, device, train_losses, it, net, loss_fn, optimizer, scheduler, log_every=log_term)
        #------------------------------------------------ Validation
        valid_acc, valid_loss = evaluate(dataloader_valid, device, net, loss_fn)
        valid_losses[it] = valid_loss
        valid_accs[it] = valid_acc
        #------------------------------------------------ Save model
        saved = ''
        if valid_acc > best_validation_acc:
            best_validation_acc = valid_acc
            saved = save_ckpt(ckpt_path, net, best_validation_acc)
        print('Epoch {} / Valid loss: {:.4f}, Valid acc: {:.4f} {}'.format(epoch, valid_loss, valid_acc, saved))
        #------------------------------------------------ One epoch end
        curr_time = time.time()
        print("One epoch time = %.2f s" %(curr_time-one_epoch_start))
        print('#------------------------------------------------------#')
    
    save_train_log(result_dir, train_losses, valid_losses, valid_accs, best_validation_acc)

    # =============================================== Test
    net, best_validation_acc = load_ckpt(ckpt_path, net)
    test_acc, test_loss = evaluate(dataloader_test, device, net, loss_fn)

    return test_acc


#%% =============================================== main
if __name__ == "__main__":
    result = []
    num_exps = len(experiment_configs)
    for i in range(num_exps):
        config = experiment_configs[i]
        print('########################################################')
        print('# Config: %s, Case: %d'\
                %(CONFIG_FNAME,config['case_num']))

        test_acc = main(config)
        
        print('# Config: %s, Case: %d, Acc: %.4f'\
                %(CONFIG_FNAME,experiment_configs[i]['case_num'],test_acc))
        print('########################################################')

        result.append([config['case_num'], test_acc])
    
    test_results = np.array(result)
    np.savetxt('%s/test_results.txt'%RESULT_ROOT_DIR, test_results, delimiter=',')

