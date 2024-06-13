import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet
from tqdm import tqdm
import time
import os
from sklearn.metrics import precision_recall_fscore_support



def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0
    all_preds = []
    all_labels = []
    start_time = time.time()
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        train_loss += loss.item() * labels.size(0)
    
    end_time = time.time()
    train_loss /= len(train_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    train_acc = (sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)) * 100
    train_time = end_time - start_time

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print("Model saved successfully.")
    
    return train_loss, train_acc, precision, recall, f1, train_time




def validate(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    valid_acc = (sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)) * 100
    valid_time = end_time - start_time
    
    return valid_acc, precision, recall, f1, valid_time




def train_and_validate(model, train_loader, valid_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1, train_time = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_acc, valid_precision, valid_recall, valid_f1, valid_time = validate(model, valid_loader, device)
        
        print(f'Epoch: {epoch + 1}/{epochs}, '
              f'Train Accuracy: {train_acc:.2f}%, '
              f'Train Loss: {train_loss:.2f}, '
              f'Train Precision: {train_precision:.2f}, '
              f'Train Recall: {train_recall:.2f}, '
              f'Train F1: {train_f1:.2f}, '
              f'Validation Accuracy: {valid_acc:.2f}%, '
              f'Validation Precision: {valid_precision:.2f}, '
              f'Validation Recall: {valid_recall:.2f}, '
              f'Validation F1: {valid_f1:.2f}, ')
        


















