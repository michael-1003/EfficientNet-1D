import os
import csv

import numpy as np
import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def read_config(fname):
    configs = []
    with open('../exp_configs/%s.csv'%fname, 'r') as config_file:
        reader = csv.DictReader(config_file)
        keys = reader.fieldnames
        for r in reader:
            r[keys[0]] = int(r[keys[0]])        # 0. case_num
                                                # 1. dataset
            r[keys[2]] = str2bool(r[keys[2]])       # 2. normalization
            r[keys[3]] = int(r[keys[3]])        # 3. batch_size
            r[keys[4]] = int(r[keys[4]])        # 4. max_epoch
                                                # 5. optimizer
            r[keys[6]] = float(r[keys[6]])      # 6. learning_rate
            r[keys[7]] = int(r[keys[7]])        # 7. lr_step
            r[keys[8]] = float(r[keys[8]])      # 8. lr_decay
            r[keys[9]] = float(r[keys[9]])      # 9. l2_decay
            r[keys[10]] = str2bool(r[keys[10]])     # 10. use_tensorboard
                                                # 11. model_name
            r[keys[12]] = float(r[keys[12]])    # 12. alpha
            r[keys[13]] = float(r[keys[13]])    # 13. beta
            r[keys[14]] = float(r[keys[14]])    # 14. gamma
            r[keys[15]] = float(r[keys[15]])    # 15. phi
                                                # 16. loss_fn
            r[keys[17]] = int(r[keys[17]])      # 17. kernel_size
            configs.append(r)

    return configs


def result_data(category, configs, results):
    all_cat = np.array([configs[i][category] for i in range(len(configs))])
    cat = np.unique(all_cat)

    return all_cat, results


def unique_average(all_x, all_y):
    x = np.unique(all_x)
    avg_y = np.zeros(x.shape)
    n = np.zeros(x.shape)
    for i in range(len(all_y)):
        idx = np.where(x==all_x[i])
        avg_y[idx] += all_y[i]
        n[idx] += 1
    avg_y /= n

    return x, avg_y



def visualize(all_x, all_y, x, y, label='', avg_line=False, xscale='linear'):
    fig = plt.figure(dpi=150, figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.plot(all_x, all_y, 'b.')
    if avg_line:
        ax.plot(x, y, 'rs-')
        for i in range(len(x)):
            ax.text(x[i], y[i]-0.2*(i%2-0.5), '%.4f'%y[i],\
                color='r', ha='center', weight='bold')
    ax.grid()

    plt.xlabel(label)
    if all_x.dtype == 'bool':
        plt.xlim([-0.5,1.5])
    plt.xscale(xscale)

    plt.ylabel('accuracy')
    plt.ylim([0,1.0])
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])

    plt.tight_layout()
