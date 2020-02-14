import torch
import torch.nn as nn

def evaluate(dataloader, device, net, loss_fn):
    net.eval()
    n = 0.
    loss = 0.
    score = 0.
    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device, dtype=torch.float)
            target = target.to(device)
            logits = net(x)
            loss += (loss_fn(logits, target).item()) * x.size(0)
            score += (logits.argmax(dim=1) == target).float().sum().item()
            n += x.size(0)
        loss /= n
        score /= n
    
    return score, loss

if __name__ == "__main__":
    pass