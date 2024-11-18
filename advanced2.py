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
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

load_dotenv()

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.toarray())
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
        
def preprocess_texts(df):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2)
    X = vectorizer.fit_transform(df['text'])
    return X, vectorizer

def prepare_target(df):
    return df['is_toxic'].astype(int)

def train_lstm(df):
    # Hiperparámetros ajustados
    input_size = 1000
    hidden_size = 128
    output_size = 1
    num_layers = 2
    batch_size = 64
    learning_rate = 0.001
    epochs = 30
    dropout = 0.2  # Reducir el dropout para evitar underfitting

    try:
        X, vectorizer = preprocess_texts(df)
        y = prepare_target(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = ImprovedLSTMModel(input_size, hidden_size, output_size, num_layers=num_layers, dropout=dropout)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)  # Paciencia mayor

        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 15  # Mayor paciencia para evitar detener demasiado temprano

        st.write("Training model...")
        progress_bar = st.progress(0)
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch.unsqueeze(1)).item()

            train_losses.append(epoch_loss / len(train_loader))
            val_losses.append(val_loss / len(test_loader))
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vectorizer': vectorizer
                }, 'lstm_model.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    st.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            progress_bar.progress((epoch + 1) / epochs)

        st.success("Training completed!")

        checkpoint = torch.load('lstm_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        y_train_pred = []
        y_test_pred = []
        y_test_prob = []

        with torch.no_grad():
            for X_batch, _ in train_loader:
                outputs = torch.sigmoid(model(X_batch))
                y_train_pred.extend((outputs > 0.5).cpu().numpy().flatten())
            for X_batch, _ in test_loader:
                outputs = torch.sigmoid(model(X_batch))
                y_test_prob.extend(outputs.cpu().numpy().flatten())
                y_test_pred.extend((outputs > 0.5).cpu().numpy().flatten())

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_test_pred)

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'learning_curves': {
                'train_losses': train_losses,
                'val_losses': val_losses
            }
        }

        pd.DataFrame([metrics]).to_json('model_metrics.json')
        return model, vectorizer, metrics

    except Exception as e:
        st.error(f"Error in training: {str(e)}")
        raise e

class YouTubeCommentScraper:
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key not found in environment variables")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def extract_video_id(self, url):
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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Accuracy", f"{metrics['train_accuracy']:.2%}")
    with col2:
        st.metric("Testing Accuracy", f"{metrics['test_accuracy']:.2%}")
    with col3:
        overfitting = metrics['train_accuracy'] - metrics['test_accuracy']
        st.metric("Model Difference", f"{overfitting:.2%}")

    st.subheader("Classification Report")
    try:
        report = metrics['classification_report']
        df_report = pd.DataFrame.from_dict(report).transpose()
        df_report = df_report.drop('support', axis=1)
        df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        st.dataframe(
            df_report.style
            .format({
                'precision': '{:.2%}',
                'recall': '{:.2%}',
                'f1-score': '{:.2%}'
            })
        )
    except Exception as e:
        st.error(f"Error displaying classification report: {str(e)}")

def analyze_comment(model, vectorizer, comment):
    model.eval()
    with torch.no_grad():
        X = vectorizer.transform([comment]).toarray()
        X_tensor = torch.FloatTensor(X)
        output = model(X_tensor)
        probability = torch.sigmoid(output).item()
    return probability

def create_streamlit_app():
    st.title("Advanced YouTube Hate Speech Detector (LSTM)")
    try:
        if os.path.exists('lstm_model.pth'):
            checkpoint = torch.load('lstm_model.pth')
            model = ImprovedLSTMModel(1000, 128, 1, num_layers=2)  # Asegúrate de que estos parámetros coincidan con el modelo guardado
            model.load_state_dict(checkpoint['model_state_dict'])
            vectorizer = checkpoint['vectorizer']
        else:
            st.info("Training new model... This may take a few minutes.")
            df = pd.read_csv('youtoxic_english_1000.csv')
            model, vectorizer, metrics = train_lstm(df)
            st.success("Model training completed!")

        scraper = YouTubeCommentScraper()

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

                            hate_comments = [r for r in results if r['is_hate']]
                            st.write(f"Found {len(hate_comments)} potentially offensive comments out of {len(results)} analyzed.")
                            for result in results:
                                if result['is_hate']:
                                    st.error(f"⚠️ {result['comment']}\nProbability: {result['probability']:.2%}")
                                else:
                                    st.success(f"✅ {result['comment']}\nProbability: {result['probability']:.2%}")

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

        if os.path.exists('model_metrics.json'):
            metrics = pd.read_json('model_metrics.json').iloc[0].to_dict()
            display_metrics(metrics)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_streamlit_app()