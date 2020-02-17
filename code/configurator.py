import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim

from models import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def read_config(fname):
    configs = []
    with open('../exp_configs/%s.csv'%fname, 'r') as config_file:
        reader = csv.DictReader(config_file)
        keys = reader.fieldnames
        for r in reader:
            r[keys[0]] = int(r[keys[0]])            # 0. case_num
                                                    # 1. dataset
            r[keys[2]] = str2bool(r[keys[2]])       # 2. normalization
            r[keys[3]] = int(r[keys[3]])            # 3. batch_size
            r[keys[4]] = int(r[keys[4]])            # 4. max_epoch
                                                    # 5. optimizer
            r[keys[6]] = float(r[keys[6]])          # 6. learning_rate
            r[keys[7]] = int(r[keys[7]])            # 7. lr_step
            r[keys[8]] = float(r[keys[8]])          # 8. lr_decay
            r[keys[9]] = float(r[keys[9]])          # 9. l2_decay
            r[keys[10]] = str2bool(r[keys[10]])     # 10. use_tensorboard
                                                    # 11. model_name
            r[keys[12]] = float(r[keys[12]])        # 12. alpha
            r[keys[13]] = float(r[keys[13]])        # 13. beta
            r[keys[14]] = float(r[keys[14]])        # 14. gamma
            r[keys[15]] = float(r[keys[15]])        # 15. phi
                                                    # 16. loss_fn
            r[keys[17]] = int(r[keys[17]])          # 17. kernel_size
            configs.append(r)

    return configs


def select_data(name):
    if name == 'nlu1':
        file_name = 'NLU/d1.csv'
        data_dim = 5000
    elif name == 'nlu2':
        file_name = 'NLU/d2.csv'
        data_dim = 1000
    elif name == 'sen1':
        file_name = ''
        data_dim = 60
    elif name == 'sen2':
        file_name = ''
        data_dim = 60
    
    return file_name, data_dim


def select_model(name, data_dim, kernel_size, num_classes, alpha, beta, phi):
    if name == 'cnn1d_adaptive':
        net = CNN1d_adaptive(kernel_size, num_classes, alpha, beta, phi)
    else: raise(ValueError('No such model named %s' % name))

    return net


def select_loss(name):
    if name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    else: raise(ValueError('No such loss named %s' % name))

    return loss_fn


def select_optimizer(name, target, learning_rate, l2_decay):
    if name == 'adam':
        optimizer = optim.Adam(target, lr=learning_rate, weight_decay=l2_decay)
    else: raise(ValueError('No such optimizer named %s' % name))

    return optimizer




if __name__ == "__main__":
    configs = read_config('exp1')

    r = configs[0]
    print(r)