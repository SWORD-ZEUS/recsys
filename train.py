import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *
from torch.utils.tensorboard import SummaryWriter
import os

def train_model(model, train_loader, valid_loader, test_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=LOG_DIR)
    best_valid_acc = 0
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user = batch['user'].to(DEVICE)
            item = batch['item'].to(DEVICE)
            genre = batch['genre'].to(DEVICE)
            tag = batch['tag'].to(DEVICE)
            rating = torch.tensor([RATING_CLASSES.index(r) for r in batch['rating']]).long().to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(user, item, genre, tag)
            loss = criterion(logits, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += rating.size(0)
            correct += (predicted == rating).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion)
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {best_valid_acc:.2f}%")
    
    writer.close()

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            user = batch['user'].to(DEVICE)
            item = batch['item'].to(DEVICE)
            genre = batch['genre'].to(DEVICE)
            tag = batch['tag'].to(DEVICE)
            rating = torch.tensor([RATING_CLASSES.index(r) for r in batch['rating']]).long().to(DEVICE)
            
            logits = model(user, item, genre, tag)
            loss = criterion(logits, rating)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += rating.size(0)
            correct += (predicted == rating).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
