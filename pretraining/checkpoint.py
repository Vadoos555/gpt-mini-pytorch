import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, loss):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss
