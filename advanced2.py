import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import requests
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configuración de dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset personalizado
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Modelo de clasificación basado en BERT
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
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# Función para entrenar el modelo
def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = None
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # Entrenamiento
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds

        # Validación
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                val_correct_preds += (preds == labels).sum().item()
                val_total_preds += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct_preds / val_total_preds

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Guardar el mejor modelo basado en precisión de validación
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

# Función para preprocesar y cargar los datos
def load_and_preprocess_data():
    # Cargar el dataset
    df = pd.read_csv('youtoxic_english_1000.csv')

    # Limpiar y preprocesar los datos
    df['Text'] = df['Text'].fillna('')  # Asegurarse de que no haya valores nulos
    df['Label'] = df['IsHatespeech'].astype(int)  # Usaremos la columna IsHatespeech como etiqueta

    # Dividir los datos en entrenamiento y validación
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Text'].tolist(), df['Label'].tolist(), test_size=0.2, random_state=42
    )

    # Tokenizar los textos
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Crear los datasets de entrenamiento y validación
    train_dataset = CommentDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = CommentDataset(val_texts, val_labels, tokenizer, max_length=128)

    # Crear los data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    return train_loader, val_loader

# Función principal
def main():
    # Cargar y preprocesar los datos
    train_loader, val_loader = load_and_preprocess_data()

    # Crear el modelo
    model = HateSpeechClassifier(n_classes=2).to(device)

    # Entrenar el modelo
    model = train_model(model, train_loader, val_loader)

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'modelo_entrenado.pth')
    st.success("Modelo entrenado y guardado correctamente.")

    # Realizar predicciones en los datos de validación (opcional)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Reporte de clasificación
    st.write(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    main()
# creo otro archivo advance_improved2.py porque este archivo tarda 15 minutos en ejecutarse con las siguientes métricas, aunque en ningún momento me pide el url:
# Epoch 1/3
# Training Loss: 0.4168, Accuracy: 0.8387
# Validation Loss: 0.4229, Accuracy: 0.8250
# Epoch 2/3
# Training Loss: 0.3332, Accuracy: 0.8612
# Validation Loss: 0.3384, Accuracy: 0.8250
# Epoch 3/3
# Training Loss: 0.2196, Accuracy: 0.9075
# Validation Loss: 0.3854, Accuracy: 0.8450