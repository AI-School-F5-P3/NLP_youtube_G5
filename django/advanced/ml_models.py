import torch
import torch.nn as nn
from transformers import BertTokenizer
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class HateDetectionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3, bidirectional=True):
        super(HateDetectionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0,
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

class HateDetectionModel:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Cargar la configuración
            config_path = os.path.join(self.models_dir, 'model_config.pkl')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"No se encontró el archivo de configuración en {config_path}")
                
            self.config = joblib.load(config_path)
            
            # Verificar que todos los parámetros necesarios estén presentes
            required_params = ['vocab_size', 'embedding_dim', 'hidden_dim', 'n_layers', 'dropout', 'bidirectional']
            missing_params = [param for param in required_params if param not in self.config]
            if missing_params:
                raise ValueError(f"Faltan los siguientes parámetros en la configuración: {missing_params}")
            
            # Inicializar modelo
            self.model = HateDetectionLSTM(
                self.config['vocab_size'],
                self.config['embedding_dim'],
                self.config['hidden_dim'],
                self.config['n_layers'],
                self.config['dropout'],
                self.config['bidirectional']
            )
            
            # Cargar pesos del modelo
            model_path = os.path.join(self.models_dir, 'lstm_model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")
                
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Cargar tokenizer
            tokenizer_path = os.path.join(self.models_dir, 'tokenizer')
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"No se encontró el tokenizer en {tokenizer_path}")
                
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            
        except Exception as e:
            logger.error(f"Error al inicializar el modelo: {str(e)}")
            raise
    
    def predict(self, text):
        try:
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
        except Exception as e:
            logger.error(f"Error en la predicción: {str(e)}")
            raise