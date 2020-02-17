import os

import torch

import matplotlib.pyplot as plt


def make_dir(root_dir, dir_name, overwrite=False):
    dir_to_make = root_dir + '/' + dir_name
    if os.path.exists(dir_to_make) and not overwrite:
        print('Result directory already exists. Delete? (y/n)')
        c = input()
        if c == 'n':
            print('Stop running')
            exit()
        elif c == 'y':
            print('Delete existing folder and try again. Otherwise, implement the file deleting code.')
    elif os.path.exists(dir_to_make) and overwrite:
        pass
    elif not os.path.exists(dir_to_make):
        os.mkdir(dir_to_make)
    
    return dir_to_make


def save_ckpt(ckpt_path, net, best_validation_acc):
    ckpt = {'net': net.state_dict(),
            'best_validation_acc':best_validation_acc}
    torch.save(ckpt,ckpt_path)
    return 'best state ckpt saved!'



def load_ckpt(ckpt_path, net):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            net.load_state_dict(ckpt['net'])
            best_validation_acc = ckpt['best_validation_acc']  
            print('Checkpoint is loaded (best_validation_acc:%.4f)' % best_validation_acc)
        except RuntimeError as e:
            print('Wrong checkpoint')
    else:
        raise(ValueError('No checkpoint exists'))
    return net, best_validation_acc



def save_train_log(result_dir, train_losses, valid_losses, valid_accs, best_validation_acc):
    #------------------------------------------------ Save train result
    with open(result_dir + '/' + 'trainloss.txt', 'w') as f:
        f.write(str(train_losses))
    with open(result_dir + '/' + 'validloss.txt', 'w') as f:
        f.write(str(valid_losses))
    with open(result_dir + '/' + 'validacc.txt', 'w') as f:
        f.write(str(valid_accs))
    
    #------------------------------------------------ Visualize train
    plt.figure(dpi=300, figsize=(6,6))
    
    plt.subplot(211)
    plt.plot(list(train_losses.keys()),list(train_losses.values()),'b',label='train loss')
    plt.plot(list(valid_losses.keys()),list(valid_losses.values()),'r',label='valid loss')
    plt.legend()
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training Curve')

    plt.subplot(212)
    plt.plot(list(valid_accs.keys()),list(valid_accs.values()),'b',label='validation accuracy')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.ylim([0,1.2])
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
    plt.title('Validation Accuracy, Best = %.4f' % best_validation_acc)
    
    plt.tight_layout()
    plt.savefig(result_dir + '/' + 'training result.png')
    plt.close('all')