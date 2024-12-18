import pandas as pd
import numpy as np
import spacy
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

# Load the spaCy language model for lemmatization
nlp = spacy.load("en_core_web_sm")

class HateSpeechDetector:
    def __init__(self):
        # Use a pipeline to standardize the process
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=1000,  # Reduced further to avoid overfitting
                ngram_range=(1, 2),
                min_df=3,  # Increased to reduce infrequent terms
                max_df=0.9,  # Adjusted to be more restrictive
                stop_words='english'
            )),
            ('dim_reduction', TruncatedSVD(n_components=100, random_state=42)),  # Dimensionality reduction
            ('classifier', LogisticRegression(
                C=0.001,  # Adjusted regularization (higher C value to avoid underfitting)
                class_weight='balanced',  # Handle class imbalance
                random_state=42,
                max_iter=1000
            ))
        ])
        
    def prepare_target(self, df):
        """Combines all target columns into a single hate label"""
        hate_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                       'IsObscene', 'IsHatespeech', 'IsRacist']
        return (df[hate_columns].sum(axis=1) > 0).astype(int)
    
    def preprocess_text(self, text):
        """Preprocess the text with lemmatization using spaCy"""
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])
    
    def train(self, df):
        """Train the model with the provided data"""
        # Preprocess the texts
        df['Processed_Text'] = df['Text'].apply(self.preprocess_text)
        
        # Prepare features and target
        X = df['Processed_Text']
        y = self.prepare_target(df)
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        ) # Stratify ensures that the class proportions in train and test sets match those of the original dataset, important for imbalanced classification problems
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate with cross-validation (10 folds for robustness)
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
        """Predict whether the text contains hate speech"""
        if isinstance(text, str):
            text = [text]
        return self.pipeline.predict_proba(text)[:, 1]
    
    def save_model(self, path):
        """Save the trained model"""
        # Create the "models" directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"The model file was not found at: {path}")
        
        detector = cls()
        detector.pipeline = joblib.load(path)
        return detector

# Streamlit Interface
def create_streamlit_app(detector, metrics):
    st.title("Hate Speech Detector")
    
    # Path of the model in the `models` folder
    model_path = os.path.join('models', 'hate_speech_model.pkl')
    
    # Load model
    try:
        detector = HateSpeechDetector.load_model(model_path)
    except FileNotFoundError:
        st.error(f"Trained model not found at {model_path}. Please train the model first.")
        return
    
    # Text area for input
    text_input = st.text_area("Enter the text to analyze:")
    
    if st.button("Analyze"):
        if text_input:
            probability = detector.predict(text_input)[0]
            st.write(f"Hate speech probability: {probability:.2%}")
            
            if probability > 0.5:
                st.error("⚠️ This text may contain hate speech.")
            else:
                st.success("✅ Este texto parece seguro.")
        else:
            st.warning("Por favor, introduce algún texto para analizar.")
            
        st.subheader("Métricas del modelo:")
        metrics_df = pd.DataFrame({
            'Métrica': ['Precisión de entrenamiento', 'Precisión de prueba', 'Promedio de validación cruzada', 'Desviación estándar de validación cruzada', 'Overfitting'],
            'Valor': [
                f"{metrics['train_accuracy']:.4f}",
                f"{metrics['test_accuracy']:.4f}",
                f"{metrics['cv_scores_mean']:.4f}",
                f"{metrics['cv_scores_std']:.4f}",
                f"{metrics['overfitting']:.4f}"
            ]
        })
        st.table(metrics_df)
    else:
        st.warning("Por favor, introduce algún texto para analizar.")

if __name__ == "__main__":
    # Cargar y entrenar el modelo
    df = pd.read_csv('youtoxic_english_1000.csv')
    detector = HateSpeechDetector()
    metrics = detector.train(df)
    print("Métricas del modelo:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Classification Report:\n{metrics['classification_report']}")
    
    model_path = os.path.join('models', 'hate_speech_model.pkl')
    print(f"Model saved at: {model_path}")
    detector.save_model(model_path)
    
    # Ejecutar la app
    create_streamlit_app(detector, metrics)
