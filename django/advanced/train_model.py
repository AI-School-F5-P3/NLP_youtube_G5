import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from transformers import BertTokenizer
import os
import logging
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class HateDetectionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3, bidirectional=True):
        super(HateDetectionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           n_layers, 
                           batch_first=True,
                           dropout=dropout if n_layers > 1 else 0,
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output[:, -1, :])
        return self.sigmoid(output), hidden
    
    def init_hidden(self, batch_size, device):
        num_directions = 2 if self.bidirectional else 1
        return (torch.zeros(self.n_layers * num_directions, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers * num_directions, batch_size, self.hidden_dim).to(device))

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
        
        encoding = self.tokenizer(text, 
                                truncation=True,
                                max_length=self.max_len,
                                padding='max_length',
                                return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            hidden = model.init_hidden(input_ids.size(0), device)
            outputs, _ = model(input_ids, hidden)
            
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
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
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
    
    # Mostrar distribución de clases
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
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
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Hiperparámetros
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    VOCAB_SIZE = tokenizer.vocab_size
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.5
    BIDIRECTIONAL = True
    
    # Usar el sampler en el DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Crear modelo
    model = HateDetectionLSTM(
        VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, BIDIRECTIONAL
    )
    model.to(device)
    
    # Configurar criterion y optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5)
    best_f1 = 0
    
    # Entrenamiento
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            hidden = model.init_hidden(input_ids.size(0), device)
            
            optimizer.zero_grad()
            outputs, hidden = model(input_ids, hidden)
            
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluación
        metrics = evaluate_model(model, test_loader, criterion, device)
        
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Metrics: {metrics}")
        
        # Actualizar learning rate
        scheduler.step(metrics['loss'])
        
        # Early stopping check
        early_stopping(metrics['loss'])
        
        # Guardar mejor modelo
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            logger.info(f"New best F1 score: {best_f1:.2f}%")
            
            # Guardar modelo
            model_path = os.path.join(MODELS_DIR, 'lstm_model.pth')
            torch.save(model.state_dict(), model_path)
            
            # Guardar configuración y métricas
            config = {
                'vocab_size': VOCAB_SIZE,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'metrics': metrics,
                'n_layers': N_LAYERS,
                'dropout': DROPOUT,
                'bidirectional': BIDIRECTIONAL
            }
            
            config_path = os.path.join(MODELS_DIR, 'model_config.pkl')
            joblib.dump(config, config_path)
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    logger.info("Training completed!")
    return metrics

if __name__ == '__main__':
    final_metrics = train_model()
    print("\nFinal Model Performance:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.2f}")