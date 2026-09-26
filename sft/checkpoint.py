import torch


def save_sft_checkpoint(path, model, optimizer, epoch, train_loss, val_loss):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    torch.save(checkpoint, path)


def load_sft_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_los']
