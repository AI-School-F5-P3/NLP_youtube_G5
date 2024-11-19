import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from googleapiclient.discovery import build
import re
import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

def preprocess_texts(df):
    """Preprocess texts using TF-IDF vectorization."""
    # Prepare hate speech target 
    hate_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                    'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist', 
                    'IsSexist', 'IsReligiousHate']
    
    # Combine hate columns and convert to binary
    df['hate_label'] = (df[hate_columns].sum(axis=1) > 0).astype(int)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, 
                                 stop_words='english', 
                                 min_df=2)
    X = vectorizer.fit_transform(df['Text']).toarray()
    
    return X, vectorizer

class TextDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
            
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(df):
    input_size = 1000
    hidden_size = 256
    output_size = 1
    batch_size = 32
    learning_rate = 0.001
    epochs = 20

    try:
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)

        # Preprocess and vectorize texts
        X, vectorizer = preprocess_texts(df)
        y = df['hate_label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, 
                                                            stratify=y)

        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = ImprovedLSTMModel(input_size, hidden_size, output_size)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                      weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                               patience=3, factor=0.5)

        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 5

        st.write("Training model...")
        progress_bar = st.progress(0)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch.view(-1, 1)).item()
            
            train_losses.append(epoch_loss / len(train_loader))
            val_losses.append(val_loss / len(test_loader))
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                # Save best model in models directory
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vectorizer': vectorizer
                }, 'models/lstm_model.pth')
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                st.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            progress_bar.progress((epoch + 1) / epochs)
            
        st.success("Training completed!")

        # Load best model for evaluation
        checkpoint = torch.load('models/lstm_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        model.eval()
        y_train_pred = []
        y_test_pred = []
        
        with torch.no_grad():
            for X_batch, _ in train_loader:
                outputs = torch.sigmoid(model(X_batch))
                y_train_pred.extend((outputs > 0.5).numpy())
            
            for X_batch, _ in test_loader:
                outputs = torch.sigmoid(model(X_batch))
                y_test_pred.extend((outputs > 0.5).numpy())

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        # Metrics visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Hate', 'Hate'],
                    yticklabels=['Non-Hate', 'Hate'])
        plt.title('Confusion Matrix')
        plt.savefig('models/confusion_matrix.png')
        plt.close()
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'learning_curves': {
                'train_losses': train_losses,
                'val_losses': val_losses
            }
        }

        # Save metrics
        pd.DataFrame([metrics]).to_json('models/model_metrics.json')
        
        return model, vectorizer, metrics
    
    except Exception as e:
        st.error(f"Error in training: {str(e)}")
        raise e

class YouTubeCommentScraper:
    def __init__(self):
        # Asegúrate de tener tu API_KEY en el archivo .env
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key not found in environment variables")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def extract_video_id(self, url):
        """Extrae el ID del video de una URL de YouTube."""
        # Maneja diferentes formatos de URL de YouTube
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith(('/embed/', '/v/')):
                return parsed_url.path.split('/')[2]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        raise ValueError("Invalid YouTube URL")

    def get_comments(self, video_url, max_comments=100):
        """Obtiene los comentarios de un video de YouTube."""
        try:
            video_id = self.extract_video_id(video_url)
            comments = []
            
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=min(max_comments, 100)
            )

            while request and len(comments) < max_comments:
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                    
                    if len(comments) >= max_comments:
                        break
                
                # Obtener la siguiente página de comentarios si existe
                if 'nextPageToken' in response and len(comments) < max_comments:
                    request = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        maxResults=min(max_comments - len(comments), 100),
                        pageToken=response['nextPageToken']
                    )
                else:
                    break
                    
            return comments

        except Exception as e:
            raise Exception(f"Error getting YouTube comments: {str(e)}")

def display_metrics(metrics):
    st.subheader("Model Performance Metrics")
    
    # Display main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Accuracy", f"{metrics['train_accuracy']:.2%}")
    with col2:
        st.metric("Testing Accuracy", f"{metrics['test_accuracy']:.2%}")
    with col3:
        overfitting = metrics['train_accuracy'] - metrics['test_accuracy']
        st.metric("Model Difference", f"{overfitting:.2%}")

def analyze_comment(model, vectorizer, comment):
    """Analiza un comentario para determinar si es ofensivo."""
    model.eval()  # Asegúrate de que el modelo esté en modo evaluación
    with torch.no_grad():
        # Preprocesar comentario
        X = vectorizer.transform([comment]).toarray()
        X_tensor = torch.FloatTensor(X)
        
        # Hacer predicción
        output = model(X_tensor)
        probability = torch.sigmoid(output).item()  # Convertir salida a probabilidad
        return probability

def create_streamlit_app():
    st.title("Advanced YouTube Hate Speech Detector (LSTM)")
    
    try:
        if os.path.exists('models/lstm_model.pth'):
            checkpoint = torch.load('models/lstm_model.pth')
            
            # Usa los mismos hiperparámetros que en train_lstm()
            model = ImprovedLSTMModel(
                input_size=1000,     # Coincide con max_features en TfidfVectorizer
                hidden_size=256,     # El tamaño de hidden_size usado en train_lstm()
                output_size=1,
                num_layers=2         # El número de capas LSTM usado
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            vectorizer = checkpoint['vectorizer']
        else:
            st.info("Training new model... This may take a few minutes.")
            df = pd.read_csv('youtoxic_english_1000.csv')
            model, vectorizer, metrics = train_lstm(df)
            st.success("Model training completed!")
        
        # Initialize YouTube scraper
        scraper = YouTubeCommentScraper()

        # YouTube URL input
        video_url = st.text_input("Enter YouTube video URL:")
        
        if st.button("Analyze Comments"):
            if video_url:
                try:
                    with st.spinner("Analyzing comments..."):
                        comments = scraper.get_comments(video_url)
                        
                        if comments:
                            results = []
                            for comment in comments:
                                probability = analyze_comment(model, vectorizer, comment)
                                results.append({
                                    'comment': comment,
                                    'probability': probability,
                                    'is_hate': probability > 0.5
                                })
                            
                            # Display results
                            hate_comments = [r for r in results if r['is_hate']]
                            st.write(f"Found {len(hate_comments)} potentially offensive comments out of {len(results)} analyzed.")
                            
                            for result in results:
                                if result['is_hate']:
                                    st.error(f"⚠️ {result['comment']}\nProbability: {result['probability']:.2%}")
                                else:
                                    st.success(f"✅ {result['comment']}\nProbability: {result['probability']:.2%}")
                            
                            # Summary metrics
                            st.subheader("Analysis Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Comments", len(comments))
                            with col2:
                                st.metric("Hate Comments", len(hate_comments))
                            with col3:
                                st.metric("Hate Percentage", f"{(len(hate_comments)/len(comments))*100:.1f}%")
                        else:
                            st.warning("No comments found in the video.")
                except Exception as e:
                    st.error(f"Error analyzing video: {str(e)}")
            else:
                st.warning("Please enter a YouTube URL.")

        # Show model metrics if available
        if os.path.exists('model_metrics.json'):
            metrics = pd.read_json('model_metrics.json').iloc[0].to_dict()
            display_metrics(metrics)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    # Show confusion matrix
    if os.path.exists('models/confusion_matrix.png'):
        st.subheader("Confusion Matrix")
        st.image('models/confusion_matrix.png', caption='Model Performance Visualization')

    # Display detailed metrics table
    if os.path.exists('models/model_metrics.json'):
        metrics = pd.read_json('models/model_metrics.json').iloc[0].to_dict()
        
        st.subheader("Detailed Performance Metrics")
        
        # Create a more comprehensive metrics display
        metrics_df = pd.DataFrame([
            {"Metric": "Training Accuracy", "Value": f"{metrics['train_accuracy']:.2%}"},
            {"Metric": "Testing Accuracy", "Value": f"{metrics['test_accuracy']:.2%}"},
            {"Metric": "Overfitting", "Value": f"{metrics['train_accuracy'] - metrics['test_accuracy']:.2%}"},
            {"Metric": "Precision (Hate Class)", "Value": f"{metrics['classification_report']['1']['precision']:.2%}"},
            {"Metric": "Recall (Hate Class)", "Value": f"{metrics['classification_report']['1']['recall']:.2%}"},
            {"Metric": "F1-Score (Hate Class)", "Value": f"{metrics['classification_report']['1']['f1-score']:.2%}"}
        ])
        
        st.table(metrics_df)

if __name__ == "__main__":
    create_streamlit_app()