import torch
from torch.utils.data import DataLoader, Dataset
from .models.transformer_model import HateDetectionTransformer
from .utils.mlflow_tracking import MLFlowTracker
from sklearn.model_selection import train_test_split
import pandas as pd

class CommentDataset(Dataset):
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
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def train_model():
    # Configurar MLFlow
    mlflow_tracker = MLFlowTracker()
    
    # Hiperparámetros
    params = {
        'learning_rate': 2e-5,
        'epochs': 3,
        'batch_size': 16,
        'max_len': 128
    }
    
    with mlflow_tracker.start_run():
        # Loguear hiperparámetros
        mlflow_tracker.log_params(params)
        
        # Cargar y preparar datos
        df = pd.read_csv('youtoxic_english_1000.csv')
        
        # Dividir datos
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'], df['toxic'], test_size=0.2
        )
        
        # Crear datasets
        train_dataset = CommentDataset(train_texts, train_labels, tokenizer)
        val_dataset = CommentDataset(val_texts, val_labels, tokenizer)
        
        # Crear dataloaders
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Entrenar modelo
        model = HateDetectionTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
        criterion = torch.nn.BCELoss()
        
        for epoch in range(params['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(outputs.squeeze(), batch['labels'])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch['input_ids'], batch['attention_mask'])
                    val_loss += criterion(outputs.squeeze(), batch['labels']).item()
                    predictions = (outputs.squeeze() > 0.5).float()
                    correct += (predictions == batch['labels']).sum().item()
                    total += len(batch['labels'])
            
            # Calcular métricas
            accuracy = correct / total
            
            # Loguear métricas
            metrics = {
                f'epoch_{epoch+1}/train_loss': train_loss / len(train_loader),
                f'epoch_{epoch+1}/val_loss': val_loss / len(val_loader),
                f'epoch_{epoch+1}/accuracy': accuracy
            }
            mlflow_tracker.log_metrics(metrics)
        
        # Guardar modelo
        mlflow_tracker.log_model(model, "hate_detection_transformer")
        
        mlflow_tracker.end_run()

if __name__ == "__main__":
    train_model()