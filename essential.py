import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import streamlit as st
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
import os

class HateSpeechDetector:
    def __init__(self):
        # Use a pipeline to standardize the process
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=800,  # Reduced number of features to reduce noise
                ngram_range=(1, 2),
                min_df=3,  # Adjusted to filter out noise
                max_df=0.9,  # More strict about frequent terms
                stop_words='english'
            )),
            ('dim_reduction', TruncatedSVD(n_components=100, random_state=42)),  # Dimensionality reduction
            ('classifier', LogisticRegression(
                C=0.05,  # Increased regularization
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ))
        ])
        
    def prepare_target(self, df):
        """Combines all the target columns into a single hate label"""
        hate_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                       'IsObscene', 'IsHatespeech', 'IsRacist']
        return (df[hate_columns].sum(axis=1) > 0).astype(int)
    
    def train(self, df):
        """Train the model with the provided data"""
        # Prepare features and target
        X = df['Text']
        y = self.prepare_target(df)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
        
        # Final predictions
        train_pred = self.pipeline.predict(X_train)
        test_pred = self.pipeline.predict(X_test)
        
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting': train_acc - test_acc,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
            'classification_report': classification_report(y_test, test_pred)
        }
    
    def predict(self, text):
        """Predict whether a text contains hate speech"""
        if isinstance(text, str):
            text = [text]
        return self.pipeline.predict_proba(text)[:, 1]
    
    def save_model(self, path):
        """Save the trained model"""
        # Create the folder if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        detector = cls()
        detector.pipeline = joblib.load(path)
        return detector

# Streamlit Interface
def create_streamlit_app(detector, metrics):
    st.title("Hate Speech Detector")
    
    # Path of the model in the `models` folder
    model_path = os.path.join('models', 'hate_speech_model.pkl')
    
    # Load the model
    try:
        detector = HateSpeechDetector.load_model(model_path)
    except FileNotFoundError:
        st.error(f"Trained model not found at {model_path}. Please train the model first.")
        return
    
    # Text input area
    text_input = st.text_area("Enter the text to analyze:")
    
    if st.button("Analyze"):
        if text_input:
            probability = detector.predict(text_input)[0]
            st.write(f"Hate speech probability: {probability:.2%}")
            
            if probability > 0.5:
                st.error("⚠️ This text may contain hate speech.")
            else:
                st.success("✅ This text seems safe.")
        else:
            st.warning("Please enter some text to analyze.")
            
        st.subheader("Model Metrics:")
        metrics_df = pd.DataFrame({
            'Metric': ['Training Accuracy', 'Test Accuracy', 'Cross-validation Mean', 'Cross-validation Std', 'Overfitting'],
            'Value': [
                f"{metrics['train_accuracy']:.4f}",
                f"{metrics['test_accuracy']:.4f}",
                f"{metrics['cv_scores_mean']:.4f}",
                f"{metrics['cv_scores_std']:.4f}",
                f"{metrics['overfitting']:.4f}"
            ]
        })
        st.table(metrics_df)
    else:
        st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    # Load and train the model
    df = pd.read_csv('youtoxic_english_1000.csv')
    detector = HateSpeechDetector()
    metrics = detector.train(df)
    print("Model metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Classification Report:\n{metrics['classification_report']}")
    
    # Save the model in the `models` folder
    model_path = os.path.join('models', 'hate_speech_model.pkl')
    print(f"Model saved at: {model_path}")
    detector.save_model(model_path)
    
    # Run the app
    create_streamlit_app(detector, metrics)
