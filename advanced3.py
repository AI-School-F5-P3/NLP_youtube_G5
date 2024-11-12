import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import requests

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
    train_dataset = CommentDataset(train_texts, train_labels, tokenizer, max_length=64)
    val_dataset = CommentDataset(val_texts, val_labels, tokenizer, max_length=64)

    # Crear los data loaders con múltiples workers para acelerar el proceso
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)

    return train_loader, val_loader

# Función para obtener comentarios de un video (esto se debe adaptar dependiendo de la fuente de datos)
def get_comments_from_video(url):
    # Aquí simulo que tenemos una función que obtiene los comentarios desde una API, como YouTube.
    # Esto es solo un ejemplo, debes integrar una API real para obtener los comentarios.
    comments = [
        "This is a nice video!",
        "I hate this, it's terrible!",
        "Amazing content, keep it up!",
        "This video should be removed, it's harmful.",
    ]
    # Asegúrate de que el formato de los comentarios se ajusta a tus necesidades.
    return comments

# Función principal
def main():
    # Pide al usuario que ingrese la URL del video
    video_url = st.text_input("Ingrese la URL del video de YouTube:")

    if video_url:
        # Obtener comentarios del video (deberías usar una API real aquí)
        comments = get_comments_from_video(video_url)
        st.write("Comentarios obtenidos:")
        st.write(comments)

        # Cargar y preprocesar los datos
        train_loader, val_loader = load_and_preprocess_data()

        # Crear el modelo
        model = HateSpeechClassifier(n_classes=2).to(device)

        # Entrenar el modelo
        model = train_model(model, train_loader, val_loader)

        # Guardar el modelo entrenado
        torch.save(model.state_dict(), 'modelo_entrenado.pth')
        st.success("Modelo entrenado y guardado correctamente.")

        # Realizar predicciones en los comentarios obtenidos del video
        model.eval()
        inputs = [comment for comment in comments]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs_enc = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
        outputs = model(inputs_enc['input_ids'], inputs_enc['attention_mask'])

        # Mostrar resultados
        _, preds = torch.max(outputs, dim=1)
        st.write("Predicciones de discurso de odio en los comentarios:")
        for comment, pred in zip(comments, preds.cpu().numpy()):
            st.write(f"Comentario: {comment} - {'Discurso de odio' if pred == 1 else 'No es discurso de odio'}")
        
if __name__ == "__main__":
    main()
