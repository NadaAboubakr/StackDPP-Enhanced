#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# # Transformer Model Architecture

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attention_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class ProteinTransformer(nn.Module):
    def __init__(self, input_dim=452, d_model=128, n_heads=8, n_layers=4, d_ff=512, 
                 max_seq_len=100, num_classes=2, dropout=0.1):
        super(ProteinTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, num_classes)
        )
        
    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Reshape to sequence format for transformer
        batch_size = x.shape[0]
        
        # Create a sequence by chunking the features
        seq_len = min(x.shape[1] // self.d_model + 1, self.positional_encoding.shape[1])
        
        # Pad or truncate input to create sequence
        if x.shape[1] % self.d_model != 0:
            padding = self.d_model - (x.shape[1] % self.d_model)
            x = torch.cat([x, torch.zeros(batch_size, padding, device=x.device)], dim=1)
        
        # Reshape to sequence format
        x = x.view(batch_size, -1, self.d_model)
        
        # Truncate if sequence is too long
        if x.shape[1] > seq_len:
            x = x[:, :seq_len, :]
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :x.shape[1], :].to(x.device)
        x = x + pos_encoding
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(x)
        
        return output

# # Utility Functions

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive metrics"""
    metrics = {
        'ACC(%)': accuracy_score(y_true, y_pred) * 100,
        'SE(%)': recall_score(y_true, y_pred) * 100,
        'SP(%)': specificity(y_true, y_pred) * 100,
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
        metrics['AP'] = average_precision_score(y_true, y_prob)
    
    return metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        pred = torch.argmax(output, dim=1)
        all_predictions.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_targets, all_predictions)
    
    return avg_loss, metrics

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            pred = torch.argmax(output, dim=1)
            prob = torch.softmax(output, dim=1)[:, 1]  # Probability of positive class
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(prob.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)
    
    return avg_loss, metrics, all_predictions, all_probabilities

# # Data Loading and Preprocessing

import os

print("Loading data...")
file_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Features', 'rf452.npz')
data = np.load(file_name)

X = data['X']
y = data['y']
test_X = data['test_X']
test_y = data['test_y']

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {test_X.shape}")
print(f"Class distribution - Training: {np.bincount(y)}")
print(f"Class distribution - Test: {np.bincount(test_y)}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled = scaler.transform(test_X)

# Convert to tensors
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.LongTensor(y)
test_X_tensor = torch.FloatTensor(test_X_scaled)
test_y_tensor = torch.LongTensor(test_y)

# # Model Training

# Hyperparameters - Updated to reduce overfitting
config = {
    'input_dim': X.shape[1],
    'd_model': 64,  # Reduced from 128
    'n_heads': 8,
    'n_layers': 2,  # Reduced from 4
    'd_ff': 256,    # Reduced from 512
    'max_seq_len': 100,
    'num_classes': 2,
    'dropout': 0.4, # Increased from 0.1
    'batch_size': 32,
    'learning_rate': 0.0001,  # Reduced from 0.001
    'epochs': 100,
    'early_stopping_patience': 15,
    'weight_decay': 1e-3  # Increased from 1e-5
}

print(f"\nModel Configuration:")
for key, value in config.items():
    print(f"{key}: {value}")

# Initialize model
model = ProteinTransformer(
    input_dim=config['input_dim'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    d_ff=config['d_ff'],
    max_seq_len=config['max_seq_len'],
    num_classes=config['num_classes'],
    dropout=config['dropout']
).to(device)

print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")

# Loss and optimizer - Using SGD with momentum for better generalization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                     momentum=0.9, weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                factor=0.5, patience=5)

# Cross-validation setup
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_results = []
fold_models = []

print(f"\nStarting {n_folds}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"\n--- Fold {fold + 1}/{n_folds} ---")
    
    # Split data
    X_train_fold = X_tensor[train_idx]
    y_train_fold = y_tensor[train_idx]
    X_val_fold = X_tensor[val_idx]
    y_val_fold = y_tensor[val_idx]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    val_dataset = TensorDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False)
    
    # Initialize model for this fold
    fold_model = ProteinTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    
    fold_optimizer = optim.SGD(fold_model.parameters(), lr=config['learning_rate'], 
                              momentum=0.9, weight_decay=config['weight_decay'])
    fold_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fold_optimizer, mode='min', 
                                                         factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_metrics = train_epoch(fold_model, train_loader, 
                                               criterion, fold_optimizer, device)
        
        # Validate
        val_loss, val_metrics, _, _ = evaluate_model(fold_model, val_loader, 
                                                    criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        fold_scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = fold_model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or patience_counter >= config['early_stopping_patience']:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['ACC(%)']:.2f}%")
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate
    fold_model.load_state_dict(best_model_state)
    _, val_metrics, _, _ = evaluate_model(fold_model, val_loader, criterion, device)
    
    cv_results.append(val_metrics)
    fold_models.append(fold_model.state_dict().copy())
    
    print(f"Fold {fold + 1} Results:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")

# Calculate cross-validation statistics
print(f"\n--- Cross-Validation Results ---")
cv_df = pd.DataFrame(cv_results)
for metric in cv_df.columns:
    mean_val = cv_df[metric].mean()
    std_val = cv_df[metric].std()
    print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

# # Final Model Training and Evaluation

print("\n--- Training Final Model on Full Training Set ---")

# Create full training dataset
full_train_dataset = TensorDataset(X_tensor, y_tensor)
full_train_loader = DataLoader(full_train_dataset, batch_size=config['batch_size'], 
                              shuffle=True)

# Test dataset
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                        shuffle=False)

# Initialize final model
final_model = ProteinTransformer(
    input_dim=config['input_dim'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    d_ff=config['d_ff'],
    max_seq_len=config['max_seq_len'],
    num_classes=config['num_classes'],
    dropout=config['dropout']
).to(device)

final_optimizer = optim.SGD(final_model.parameters(), lr=config['learning_rate'], 
                           momentum=0.9, weight_decay=config['weight_decay'])
final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', 
                                                      factor=0.5, patience=5)

# Training loop for final model
best_train_loss = float('inf')
patience_counter = 0

for epoch in range(config['epochs']):
    train_loss, train_metrics = train_epoch(final_model, full_train_loader, 
                                          criterion, final_optimizer, device)
    
    final_scheduler.step(train_loss)
    
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        patience_counter = 0
        best_final_model_state = final_model.state_dict().copy()
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_metrics['ACC(%)']:.2f}%")
    
    if patience_counter >= config['early_stopping_patience']:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best final model and evaluate on test set
final_model.load_state_dict(best_final_model_state)

print("\n--- Final Test Results ---")
test_loss, test_metrics, test_predictions, test_probabilities = evaluate_model(
    final_model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# # Save Results and Model

# Save results
results_df = pd.DataFrame([test_metrics])
results_df['Model'] = 'Transformer'
results_df.to_csv('Report_Transformer_test.csv', index=False)

# Save cross-validation results
cv_df['Model'] = 'Transformer'
cv_df.to_csv('Report_Transformer_CV.csv', index=False)

# Save model
torch.save({
    'model_state_dict': final_model.state_dict(),
    'config': config,
    'test_metrics': test_metrics,
    'cv_results': cv_results
}, 'transformer_model.pth')

print(f"\nResults saved to:")
print("- Report_Transformer_test.csv")
print("- Report_Transformer_CV.csv") 
print("- transformer_model.pth")

print(f"\n--- Summary ---")
print(f"Traditional ML Best (SVC tuned): ~92% accuracy")
print(f"Transformer Model: {test_metrics['ACC(%)']:.2f}% accuracy")
print(f"Improvement: {test_metrics['ACC(%)'] - 92:.2f}%")

# # Prediction Function for New Sequences

def predict_sequence(model, features, scaler, device):
    """
    Predict DNA binding for a new sequence
    
    Args:
        model: Trained transformer model
        features: Feature vector (numpy array)
        scaler: Fitted StandardScaler
        device: torch device
    
    Returns:
        prediction: 0 (Non-binding) or 1 (Binding)
        probability: Probability of binding
    """
    model.eval()
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    with torch.no_grad():
        output = model(features_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].cpu().item()
        pred = torch.argmax(output, dim=1)[0].cpu().item()
    
    return pred, prob

print(f"\nTransformer model training complete!")
print(f"Model can be loaded using: torch.load('transformer_model.pth')")

