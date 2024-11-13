import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from transformers import BertTokenizer
import os

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

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            hidden = model.init_hidden(input_ids.size(0), device)
            outputs, _ = model(input_ids, hidden)
            
            predicted = (outputs.squeeze() > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
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
    csv_path = os.path.join(os.path.dirname(BASE_DIR), 'youtoxic_english_1000.csv')
    df = pd.read_csv(csv_path)
    
    toxic_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                     'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist', 
                     'IsSexist', 'IsReligiousHate']
    
    df['is_toxic'] = df[toxic_columns].any(axis=1).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'].values, 
        df['is_toxic'].values,
        test_size=0.2,
        random_state=42
    )
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_path = os.path.join(MODELS_DIR, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    
    train_dataset = HateCommentDataset(X_train, y_train, tokenizer)
    test_dataset = HateCommentDataset(X_test, y_test, tokenizer)
    
    # Hiperparámetros ajustados
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0005
    VOCAB_SIZE = tokenizer.vocab_size
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HateDetectionLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, BIDIRECTIONAL)
    model.to(device)
    
    # Cargar state_dict si existe
    model_path = os.path.join(MODELS_DIR, 'lstm_model.pth')
    if os.path.exists(model_path):
        print(f"Cargando pesos desde {model_path}")
        try:
            model.load_state_dict(torch.load(model_path), strict=True)
        except RuntimeError as e:
            print("Error al cargar los pesos:", e)
            print("Ignorando los pesos no coincidentes y cargando los parámetros existentes...")
            model.load_state_dict(torch.load(model_path), strict=False)
        model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_metrics = None
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            hidden = model.init_hidden(input_ids.size(0), device)
            
            optimizer.zero_grad()
            outputs, hidden = model(input_ids, hidden)
            
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        
        if (epoch + 1) % 2 == 0:
            metrics = evaluate_model(model, test_loader, device)
            print(f'Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'Precision: {metrics["precision"]:.2f}%')
            print(f'Recall: {metrics["recall"]:.2f}%')
            print(f'F1 Score: {metrics["f1_score"]:.2f}%')
            
            if best_metrics is None or metrics['f1_score'] > best_metrics['f1_score']:
                best_metrics = metrics
                torch.save(model.state_dict(), model_path)
                
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

if __name__ == '__main__':
    train_model()
