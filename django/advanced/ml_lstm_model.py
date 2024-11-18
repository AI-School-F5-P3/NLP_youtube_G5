import torch
import torch.nn as nn
import joblib
import os
import logging
from torchtext.data import get_tokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class LSTMPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 128
        self.load_model()
        
    def load_model(self):
        try:
            # Obtener directorio base
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'models')
            
            # Cargar configuración
            config = joblib.load(os.path.join(models_dir, 'lstm_config.pkl'))
            
            # Inicializar modelo
            self.model = LSTMHateDetectionModel(
                vocab_size=config['vocab_size'],
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                n_layers=config['n_layers'],
                dropout=config['dropout']
            ).to(self.device)
            
            # Cargar pesos del modelo
            self.model.load_state_dict(torch.load(
                os.path.join(models_dir, 'lstm_model.pth'),
                map_location=self.device
            ))
            self.model.eval()
            
            # Cargar tokenizer y vocabulario
            self.tokenizer = joblib.load(os.path.join(models_dir, 'lstm_tokenizer.pkl'))
            self.vocab = torch.load(os.path.join(models_dir, 'lstm_vocab.pth'))
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def preprocess_text(self, text):
        tokens = self.tokenizer(str(text))
        ids = [self.vocab[token] for token in tokens]
        
        if len(ids) < self.max_len:
            ids = ids + [self.vocab['<pad>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
            
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
    def predict(self, text):
        try:
            # Preprocesar texto
            text_tensor = self.preprocess_text(text)
            
            # Realizar predicción
            with torch.no_grad():
                output = self.model(text_tensor).squeeze()
                probability = float(output.item())
                prediction = probability > 0.5
                
            return {
                'is_toxic': bool(prediction),
                'probability': probability * 100  # Convertir a porcentaje
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'is_toxic': False,
                'probability': 0.0,
                'error': str(e)
            }

    def predict_batch(self, texts, batch_size=32):
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_tensors = [self.preprocess_text(text) for text in batch_texts]
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor).squeeze()
                probabilities = outputs.cpu().numpy()
                predictions = probabilities > 0.5
                
            batch_results = [
                {
                    'is_toxic': bool(pred),
                    'probability': float(prob) * 100
                }
                for pred, prob in zip(predictions, probabilities)
            ]
            results.extend(batch_results)
            
        return results

# Función de utilidad para uso directo
def get_lstm_predictor():
    return LSTMPredictor()