import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

# Cargar los datos
try:
    data = pd.read_csv('youtoxic_english_1000.csv')
    print("Datos cargados correctamente.")
except Exception as e:
    print(f"Error al cargar los datos: {e}")

# Crear la columna combinada `IsHateSpeech`
hate_columns = [
    "IsToxic", "IsAbusive", "IsThreat", "IsProvocative", "IsObscene", 
    "IsHatespeech", "IsRacist", "IsNationalist", "IsSexist", 
    "IsHomophobic", "IsReligiousHate", "IsRadicalism"
]
data['IsHateSpeech'] = data[hate_columns].any(axis=1)

# Definir textos y etiquetas combinadas
texts = data["Text"]
labels = data["IsHateSpeech"].astype(int)  # Convierte a 0 y 1 para el modelo

# Codificar etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Definir el dataset personalizado
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Crear el modelo
class HateSpeechClassifier(nn.Module):
    def __init__(self, n_classes):
        super(HateSpeechClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output  # pooler_output es un tensor
        output = self.drop(pooled_output)
        return self.out(output)

# Tokenizador y parámetros del modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = HateSpeechClassifier(n_classes=len(label_encoder.classes_))  # Usa el número de clases correcto

# Configura el dispositivo en CPU para evitar problemas de GPU
device = torch.device("cpu")
model = model.to(device)
print("Modelo cargado y enviado al dispositivo.")

# Dividir los datos y preparar DataLoaders
X_train, X_val, y_train, y_val = train_test_split(texts, y, test_size=0.2, random_state=42)
train_dataset = CommentDataset(X_train.tolist(), y_train, tokenizer, max_length=160)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print("Datos de entrenamiento preparados.")

# Optimización y función de pérdida
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss().to(device)

# Entrenamiento básico
for epoch in range(3):  # Ajusta los epochs según sea necesario
    model.train()
    print(f"Epoch {epoch+1} comenzando...")
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:  # Imprime cada 10 iteraciones para seguimiento
            print(f"Batch {i}, Loss: {loss.item()}")
    
    print(f"Epoch {epoch+1} completado. Última pérdida: {loss.item()}")

print("Entrenamiento completado.")
