# advanced/train_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from transformers import BertTokenizer, BertModel
import os
import logging
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Definir modelo de clasificación con BERT
class BertHateDetectionModel(nn.Module):
    def __init__(self, hidden_dim=768, dropout=0.3):
        super(BertHateDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.sigmoid(self.fc(pooled_output))

# Dataset personalizado
class HateCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            predicted = (outputs.squeeze() > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_labels, all_predictions) * 100,
        'precision': precision_score(all_labels, all_predictions, zero_division=0) * 100,
        'recall': recall_score(all_labels, all_predictions, zero_division=0) * 100,
        'f1_score': f1_score(all_labels, all_predictions, zero_division=0) * 100
    }
    
    return metrics

def train_model():
    # Cargar y preparar datos
    logger.info("Loading dataset...")
    csv_path = os.path.join(os.path.dirname(BASE_DIR), 'youtoxic_english_1000.csv')
    df = pd.read_csv(csv_path)
    
    toxic_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                    'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist', 
                    'IsSexist', 'IsReligiousHate']
    
    df['is_toxic'] = df[toxic_columns].any(axis=1).astype(int)
    X = df['Text'].values
    y = df['is_toxic'].values
    
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Configurar tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_path = os.path.join(MODELS_DIR, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    
    # Crear datasets
    train_dataset = HateCommentDataset(X_train, y_train, tokenizer)
    test_dataset = HateCommentDataset(X_test, y_test, tokenizer)
    
    # Calcular pesos para el sampling
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Configuración de entrenamiento
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = BertHateDetectionModel()
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Entrenamiento
    best_f1 = 0
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        metrics = evaluate_model(model, test_loader, criterion, device)
        
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Metrics: {metrics}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            logger.info(f"New best F1 score: {best_f1:.2f}%")
            
            model_path = os.path.join(MODELS_DIR, 'bert_model.pth')
            torch.save(model.state_dict(), model_path)
            
            config = {
                'hidden_dim': 768,
                'dropout': 0.3,
                'max_len': 128,
                'metrics': metrics
            }
            config_path = os.path.join(MODELS_DIR, 'model_config.pkl')
            joblib.dump(config, config_path)
    
    logger.info("Training completed!")
    return metrics

if __name__ == '__main__':
    final_metrics = train_model()
    print("\nFinal Model Performance:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.2f}")
