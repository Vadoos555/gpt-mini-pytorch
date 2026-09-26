import os
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.model_config import GPTConfig
from data.dataset import GPTDataset
from model.gpt import GPT
from pretraining.checkpoint import save_checkpoint, load_checkpoint


# -------------------------
# 1. Data
# -------------------------

tokens = [
    10, 20, 30, 40, 50,
    60, 70, 80, 90, 100,
    110, 120, 130, 140, 150,
    160, 170, 180, 190, 200
]

split = int(len(tokens) * 0.8)

train_tokens = tokens[:split]
val_tokens = tokens[split:]

train_dataset = GPTDataset(train_tokens, context_size=3)
val_dataset = GPTDataset(val_tokens, context_size=3)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# -------------------------
# 2. Model
# -------------------------

config = GPTConfig()

model = GPT(config)

# -------------------------
# 3. Loss
# -------------------------

loss_fn = nn.CrossEntropyLoss()

# -------------------------
# 4. Optimizer
# -------------------------

optimazer = torch.optim.AdamW(model.parameters(), lr=0.0003)

# -------------------------
# 5. Scheduler
# -------------------------

scheduler = torch.optim.lr_scheduler.StepLR(optimazer, step_size=2, gamma=0.5)

# -------------------------
# 6. CHECKPOINT DIRECTORY
# -------------------------

os.makedirs('checkpoints', exist_ok=True)

# -------------------------
# 7. Training
# -------------------------

num_epochs = 5

best_val_loss = float('inf')

for epoch in range(num_epochs):
    
    model.train()
    
    total_train_loss = 0
    num_train_batches = 0
    
    for x, y in train_loader:
        
        # Clear old gradients
        optimazer.zero_grad()
        
        # Forward pass
        logits = model(x)
        
        # reshape [B, T, V] --> [B*T, V]
        logits = logits.view(-1, logits.size(-1))
        targets = y.view(-1)
        
        # Calculate loss
        loss = loss_fn(logits, targets)
        
        # Backpropagation
        loss.backward()
        
        # Update weights
        optimazer.step()
        
        # Save loss for epoch average
        total_train_loss += loss.item()
        num_train_batches += 1
    
    avg_train_loss = total_train_loss / num_train_batches
        
    
    # -------------------------
    # 8. Validation
    # -------------------------

    model.eval()

    total_val_los = 0.0
    num_batches = 0

    with torch.no_grad():
        
        for x, y in val_loader:
            
            logits = model(x)
            logits = logits.view(-1, logits.size(-1))
            
            targets = y.view(-1)
            
            val_loss = loss_fn(logits, targets)
            
            total_val_los += val_loss.item()
            num_batches += 1

    avg_val_loss = total_val_los / num_batches
    
    val_ppl = math.exp(avg_val_loss)


    # ----------------------------
    # 9. Scheduler
    # ----------------------------

    current_lr = optimazer.param_groups[0]['lr']

    scheduler.step()

    # -------------------------
    # 10. Output
    # -------------------------

    print(
        f"epoch {epoch + 1}, "
        f"train loss = {avg_train_loss:.4f}, "
        f"val loss = {avg_val_loss:.4f}, "
        f"val ppl = {val_ppl:.2f}, "
        f"lr = {current_lr:.7f}"
    )
    
    # -------------------------
    # 11. Save Chechpoint
    # -------------------------
    
    checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch + 1}.pt'
    best_path = 'checkpoints/best.pt'
    
    save_checkpoint(checkpoint_path, model, optimazer, scheduler, epoch+1, avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        save_checkpoint(best_path, model, optimazer, scheduler, epoch+1, avg_val_loss)
        
        print(
            f'  --> new best model: '
            f'val_loss = {best_val_loss: .4f}'
        )
    