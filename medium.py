import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import streamlit as st
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import re
import googleapiclient.discovery

class YouTubeCommentScraper:
    def __init__(self):
        self.api_key = 'AIzaSyAnz3_21wonjoG6DuwwAIbdyius7f955jk'
        self.youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=self.api_key)

    def extract_video_id(self, url):
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_comments(self, video_url, max_comments=100):
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("URL de video inválida")

        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_comments
            )
            response = request.execute()

            comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]]
            return comments
        except Exception as e:
            raise Exception(f"Error al obtener los comentarios del video: {str(e)}")

class HateSpeechDetector:
    def __init__(self):
        # Crear clasificadores base
        clf1 = LogisticRegression(C=0.1, class_weight='balanced', random_state=42)
        clf2 = MultinomialNB(alpha=0.1)
        clf3 = SVC(C=0.1, class_weight='balanced', random_state=42, probability=True)

        # Crear ensemble
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
            ('scaler', StandardScaler(with_mean=False)),
            ('ensemble', VotingClassifier(
                estimators=[
                    ('lr', clf1),
                    ('nb', clf2),
                    ('svc', clf3)
                ],
                voting='soft'
            ))
        ])
        self.scraper = YouTubeCommentScraper()

    def prepare_target(self, df):
        hate_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative',
                       'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist',
                       'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
        return (df[hate_columns].sum(axis=1) > 0).astype(int)

    def train(self, df):
        X = df['Text']
        y = self.prepare_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline.fit(X_train, y_train)

        train_pred = self.pipeline.predict(X_train)
        test_pred = self.pipeline.predict(X_test)

        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting': train_acc - test_acc,
            'classification_report': classification_report(y_test, test_pred)
        }

    def analyze_video(self, video_url):
        try:
            comments = self.scraper.get_comments(video_url)
            predictions = self.predict(comments)

            results = []
            for comment, pred_prob in zip(comments, predictions):
                results.append({
                    'comment': comment,
                    'hate_probability': pred_prob,
                    'is_hate': pred_prob > 0.5
                })

            return results
        except Exception as e:
            raise Exception(f"Error al analizar el video: {str(e)}")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict_proba(texts)[:, 1]

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load_model(cls, path):
        detector = cls()
        with open(path, 'rb') as f:
            detector.pipeline = pickle.load(f)
        return detector

def create_streamlit_app():
    st.title("Detector de Mensajes de Odio")

    try:
        detector = HateSpeechDetector.load_model('hate_speech_model_enhanced.pkl')
    except:
        st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
        return

    tab1, tab2 = st.tabs(["Analizar Texto", "Analizar Video de YouTube"])

    with tab1:
        text_input = st.text_area("Introduce el texto a analizar:")
        if st.button("Analizar Texto"):
            if text_input:
                probability = detector.predict(text_input)[0]
                st.write(f"Probabilidad de contenido de odio: {probability:.2%}")

                if probability > 0.5:
                    st.error("⚠️ Este texto puede contener mensajes de odio.")
                else:
                    st.success("✅ Este texto parece seguro.")

    with tab2:
        video_url = st.text_input("Introduce la URL del video de YouTube:")
        if st.button("Analizar Video"):
            if video_url:
                with st.spinner("Analizando comentarios..."):
                    try:
                        results = detector.analyze_video(video_url)

                        hate_comments = [r for r in results if r['is_hate']]
                        st.write(f"Se encontraron {len(hate_comments)} comentarios potencialmente ofensivos de {len(results)} analizados.")

                        for result in results:
                            if result['is_hate']:
                                st.error(f"⚠️ {result['comment']}\nProbabilidad: {result['hate_probability']:.2%}")
                            else:
                                st.success(f"✅ {result['comment']}\nProbabilidad: {result['hate_probability']:.2%}")
                    except Exception as e:
                        st.error(f"Error al analizar el video: {str(e)}")

if __name__ == '__main__':
    create_streamlit_app()