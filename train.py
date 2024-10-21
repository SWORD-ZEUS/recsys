import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *

def train_model(model, train_loader, test_loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user = batch['user'].to(DEVICE)
            item = batch['item'].to(DEVICE)
            genre = batch['genre'].to(DEVICE)
            rating = batch['rating'].to(DEVICE)
            tag = batch['tag'].to(DEVICE)
            
            optimizer.zero_grad()
            output = model(user, item, genre, tag)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if EVALUATE_WHILE_TRAINING:
            evaluate_model(model, test_loader)

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(DEVICE)
            item = batch['item'].to(DEVICE)
            genre = batch['genre'].to(DEVICE)
            tag = batch['tag'].to(DEVICE)
            rating = batch['rating'].to(DEVICE) 
            
            output = model(user, item, genre, tag)
            loss = criterion(output, rating)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

