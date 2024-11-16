# advanced/ml_models.py
import torch
import torch.nn as nn
from transformers import BertTokenizer
import joblib
import os
import logging
from .train_model import BertHateDetectionModel

logger = logging.getLogger(__name__)

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
            
            # Inicializar el modelo de BERT
            self.model = BertHateDetectionModel(
                hidden_dim=self.config.get('hidden_dim', 768),
                dropout=self.config.get('dropout', 0.3)
            )
            
            # Cargar los pesos del modelo
            model_path = os.path.join(self.models_dir, 'bert_model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")
                
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Cargar el tokenizer
            tokenizer_path = os.path.join(self.models_dir, 'tokenizer')
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"No se encontró el tokenizer en {tokenizer_path}")
                
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            
        except Exception as e:
            logger.error(f"Error al inicializar el modelo: {str(e)}")
            raise
    
    def predict(self, text):
        try:
            # Preprocesar el texto
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_ids, attention_mask)
                probability = prediction.item()
                is_toxic = probability > 0.5
                
            return {
                'is_toxic': is_toxic,
                'probability': probability  # Cambiado de 'confidence' a 'probability'
            }
        except Exception as e:
            logger.error(f"Error en la predicción: {str(e)}")
            raise