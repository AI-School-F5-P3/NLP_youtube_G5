import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import logging
from tqdm import tqdm
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class LSTMHateDetectionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.3):
        super(LSTMHateDetectionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           n_layers, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.sigmoid(self.fc(hidden))

class HateCommentDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        tokens = self.tokenizer(text)
        ids = [self.vocab[token] for token in tokens]
        
        if len(ids) < self.max_len:
            ids = ids + [self.vocab['<pad>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
            
        return {
            'text': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train_lstm_model():
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
    
    # Configurar tokenizer y vocabulario
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(str(text))
    
    vocab = build_vocab_from_iterator(yield_tokens(X),
                                    specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    # Guardar tokenizer y vocab
    joblib.dump(tokenizer, os.path.join(MODELS_DIR, 'lstm_tokenizer.pkl'))
    torch.save(vocab, os.path.join(MODELS_DIR, 'lstm_vocab.pth'))
    
    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear datasets
    train_dataset = HateCommentDataset(X_train, y_train, vocab, tokenizer)
    test_dataset = HateCommentDataset(X_test, y_test, vocab, tokenizer)
    
    # Configuración de entrenamiento
    BATCH_SIZE = 32
    EPOCHS = 10
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMHateDetectionModel(
        len(vocab), 
        EMBEDDING_DIM, 
        HIDDEN_DIM, 
        N_LAYERS, 
        DROPOUT
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Entrenamiento
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}'):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            predictions = model(texts).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Evaluación
        model.eval()
        predictions = []
        actual = []
        
        with torch.no_grad():
            for batch in test_loader:
                texts = batch['text'].to(device)
                labels = batch['label']
                
                outputs = model(texts).squeeze()
                predicted = (outputs > 0.5).cpu().numpy()
                
                predictions.extend(predicted)
                actual.extend(labels.numpy())
        
        accuracy = accuracy_score(actual, predictions)
        precision = precision_score(actual, predictions)
        recall = recall_score(actual, predictions)
        f1 = f1_score(actual, predictions)
        
        logger.info(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
        logger.info(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}, F1: {f1:.4f}')
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'lstm_model.pth'))
            
            config = {
                'vocab_size': len(vocab),
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'n_layers': N_LAYERS,
                'dropout': DROPOUT
            }
            joblib.dump(config, os.path.join(MODELS_DIR, 'lstm_config.pkl'))

if __name__ == '__main__':
    train_lstm_model()