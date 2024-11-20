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
import joblib
from sklearn.pipeline import Pipeline
import re
import googleapiclient.discovery
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class YouTubeCommentScraper:
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
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
            raise ValueError("Invalid video URL")

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
            raise Exception(f"Error getting video comments: {str(e)}")

class HateSpeechDetector:
    def __init__(self):
        # Create base classifiers
        clf1 = LogisticRegression(C=0.05, class_weight='balanced', random_state=42)
        clf2 = MultinomialNB(alpha=0.1)
        clf3 = SVC(C=0.1, class_weight='balanced', random_state=42, probability=True)

        # Create ensemble
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
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
        self.metrics = None  # Store metrics for later use

    def prepare_target(self, df):
        hate_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative',
                       'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist',
                       'IsSexist', 'IsReligiousHate']
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
        
        self.metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting': train_acc - test_acc,
            'classification_report': classification_report(y_test, test_pred)
        }
        
        return self.metrics

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
            raise Exception(f"Error analyzing video: {str(e)}")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict_proba(texts)[:, 1]

    def save_model(self, path):
        # Create the models directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.pipeline, path)
        
        # Save metrics in the same directory
        metrics_path = os.path.join(os.path.dirname(path), 'model_metrics.joblib')
        joblib.dump(self.metrics, metrics_path)
        
        print(f"Model saved at: {path}")
        print(f"Metrics saved at: {metrics_path}")

    @classmethod
    def load_model(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The model file was not found at: {path}")
        
        detector = cls()
        detector.pipeline = joblib.load(path)
        
        # Load metrics if they exist
        metrics_path = os.path.join(os.path.dirname(path), 'model_metrics.joblib')
        try:
            detector.metrics = joblib.load(metrics_path)
        except:
            detector.metrics = None
            
        return detector

def create_streamlit_app():
    st.title("Hate Speech Detector")

    # Define model path in the models directory
    model_path = os.path.join('models', 'hate_speech_model.joblib')

    try:
        detector = HateSpeechDetector.load_model(model_path)
    except FileNotFoundError:
        st.error(f"No trained model found at {model_path}. Please train the model first.")
        return

    tab1, tab2 = st.tabs(["Analyze Text", "Analyze YouTube Video"])

    with tab1:
        text_input = st.text_area("Enter text to analyze:")
        if st.button("Analyze Text"):
            if text_input:
                probability = detector.predict(text_input)[0]
                st.write(f"Hate speech probability: {probability:.2%}")

                if probability > 0.5:
                    st.error("⚠️ This text may contain hate speech.")
                else:
                    st.success("✅ This text appears to be safe.")

                # Display metrics if available
                if detector.metrics:
                    st.subheader("Model Metrics:")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Training Accuracy', 'Test Accuracy', 'Overfitting'],
                        'Value': [
                            f"{detector.metrics['train_accuracy']:.4f}",
                            f"{detector.metrics['test_accuracy']:.4f}",
                            f"{detector.metrics['overfitting']:.4f}"
                        ]
                    })
                    st.table(metrics_df)

    with tab2:
        video_url = st.text_input("Enter YouTube video URL:")
        if st.button("Analyze Video"):
            if video_url:
                with st.spinner("Analyzing comments..."):
                    try:
                        results = detector.analyze_video(video_url)

                        hate_comments = [r for r in results if r['is_hate']]
                        st.write(f"Found {len(hate_comments)} potentially offensive comments out of {len(results)} analyzed.")

                        for result in results:
                            if result['is_hate']:
                                st.error(f"⚠️ {result['comment']}\nProbability: {result['hate_probability']:.2%}")
                            else:
                                st.success(f"✅ {result['comment']}\nProbability: {result['hate_probability']:.2%}")

                        # Display metrics if available
                        if detector.metrics:
                            st.subheader("Model Metrics:")
                            metrics_df = pd.DataFrame({
                                'Metric': ['Training Accuracy', 'Test Accuracy', 'Overfitting'],
                                'Value': [
                                    f"{detector.metrics['train_accuracy']:.4f}",
                                    f"{detector.metrics['test_accuracy']:.4f}",
                                    f"{detector.metrics['overfitting']:.4f}"
                                ]
                            })
                            st.table(metrics_df)
                            
                    except Exception as e:
                        st.error(f"Error analyzing video: {str(e)}")

# Training the model and saving the .joblib file
if __name__ == "__main__":
    # Load training data
    df = pd.read_csv('youtoxic_english_1000.csv')
    
    # Create and train the hate speech detector
    detector = HateSpeechDetector()
    metrics = detector.train(df)
    
    print("Model metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Classification Report:\n{metrics['classification_report']}")
    
    # Save the trained model in the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'hate_speech_model.joblib')
    detector.save_model(model_path)

    # Now you can load and use the saved model in the Streamlit interface
    create_streamlit_app()