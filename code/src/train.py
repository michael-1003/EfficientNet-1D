import torch
import torch.nn as nn


def train_1epoch(dataloader, device, train_losses, iteration, net, loss_fn, optimizer, scheduler, log_every=5):
    net.train()
    for x, target in dataloader:
        iteration += 1
        #------------------------ Inputs to device
        x = x.to(device, dtype=torch.float)
        target = target.to(device)
        #------------------------ Feed data into the network and get outputs
        logits = net(x)
        #------------------------ Compute loss
        loss = loss_fn(logits,target)
        #------------------------ Flush gradients
        optimizer.zero_grad()
        #------------------------ Back propagtion
        loss.backward()
        #------------------------ Update optimizer
        optimizer.step()
        if iteration % log_every == 0:
            print('- Iter:{} / Train loss: {:.4f}'.format(iteration, loss.item()))
            train_losses[iteration] = loss.item()
    #------------------------ Update learning rate
    scheduler.step()

    return train_losses, iteration, net, optimizer, scheduler
    

#%% =============================================== main
if __name__ == "__main__":
    pass
