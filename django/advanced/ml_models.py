# advanced/ml_models.py
from django.conf import settings
import torch
import torch.nn as nn
from transformers import BertTokenizer
import joblib
import os

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
        output = self.fc(output[:, -1, :])
        return self.sigmoid(output), hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, 128).to(device),
                torch.zeros(2, batch_size, 128).to(device))

class HateDetectionModel:
    def __init__(self):
        # Usar la ruta correcta para el directorio de modelos
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar la configuración del modelo con joblib
        config_path = os.path.join(self.models_dir, 'model_config.pkl')
        with open(config_path, 'rb') as f:
            config = joblib.load(f)
        
        # Inicializar modelo con la configuración guardada
        self.model = HateDetectionLSTM(
            config['vocab_size'],
            config['embedding_dim'],
            config['hidden_dim']
        )
        
        # Cargar pesos del modelo
        model_path = os.path.join(self.models_dir, 'lstm_model.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Cargar tokenizer
        tokenizer_path = os.path.join(self.models_dir, 'tokenizer')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    def predict(self, text):
        # Preprocesar texto
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        
        with torch.no_grad():
            hidden = self.model.init_hidden(1, self.device)
            prediction, _ = self.model(input_ids, hidden)
            is_toxic = prediction.item() > 0.5
            
        return {
            'is_toxic': is_toxic,
            'confidence': prediction.item()
        }