# advanced/train_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from transformers import BertTokenizer
import os

# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class HateDetectionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.2):
        super(HateDetectionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output[:, -1, :])  # Tomar solo el último estado oculto
        return self.sigmoid(output), hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, 128).to(device),
                torch.zeros(2, batch_size, 128).to(device))

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

def train_model():
    # Crear directorio para modelos si no existe
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # Cargar y preparar datos
    # Actualizar la ruta del CSV para que sea relativa al directorio actual
    csv_path = os.path.join(os.path.dirname(BASE_DIR), 'youtoxic_english_1000.csv')
    df = pd.read_csv(csv_path)
    
    # Combinar diferentes tipos de toxicidad
    toxic_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                    'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist', 
                    'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
    
    df['is_toxic'] = df[toxic_columns].any(axis=1).astype(int)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'].values, 
        df['is_toxic'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Inicializar tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Guardar tokenizer para uso posterior
    tokenizer_path = os.path.join(MODELS_DIR, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    
    # Crear datasets
    train_dataset = HateCommentDataset(X_train, y_train, tokenizer)
    test_dataset = HateCommentDataset(X_test, y_test, tokenizer)
    
    # Parámetros
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    VOCAB_SIZE = tokenizer.vocab_size
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modelo
    model = HateDetectionLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    model.to(device)
    
    # Criterio y optimizador
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Obtener batch
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Inicializar hidden state
            hidden = model.init_hidden(input_ids.size(0), device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, hidden = model(input_ids, hidden)
            
            # Calcular pérdida
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Imprimir progreso
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')
        
        # Evaluación
        if (epoch + 1) % 2 == 0:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    
                    hidden = model.init_hidden(input_ids.size(0), device)
                    outputs, _ = model(input_ids, hidden)
                    
                    predicted = (outputs.squeeze() > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Accuracy: {accuracy:.2f}%')
    
    # Guardar modelo con ruta absoluta
    model_path = os.path.join(MODELS_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Guardar configuración del modelo
    config = {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM
    }
    
    # Guardar la configuración del modelo con joblib
    config_path = os.path.join(MODELS_DIR, 'model_config.pkl')
    joblib.dump(config, config_path)

if __name__ == '__main__':
    train_model()