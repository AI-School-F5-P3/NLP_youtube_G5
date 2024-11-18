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

# Cargar el modelo de lenguaje de spacy para lematización
nlp = spacy.load("en_core_web_sm")

class HateSpeechDetector:
    def __init__(self):
        # Usar pipeline para estandarizar el proceso
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=1000,  # Reducido aún más para evitar overfitting
                ngram_range=(1, 1),
                min_df=5,  # Aumentado para reducir términos poco frecuentes
                max_df=0.75,  # Ajustado para ser más restrictivo
                stop_words='english'
            )),
            ('classifier', LogisticRegression(
                C=0.001,  # Regularización ajustada (un valor de C mayor que 0.01 para evitar underfitting)
                class_weight='balanced',  # Manejar desbalance de clases
                random_state=42,
                max_iter=1000
            ))
        ])
        
    def prepare_target(self, df):
        """Combina todas las columnas objetivo en una sola etiqueta de odio"""
        hate_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                       'IsObscene', 'IsHatespeech', 'IsRacist']
        return (df[hate_columns].sum(axis=1) > 0).astype(int)
    
    def preprocess_text(self, text):
        """Preprocesa el texto con lematización usando spaCy"""
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])
    
    def train(self, df):
        """Entrena el modelo con los datos proporcionados"""
        # Preprocesar los textos
        df['Processed_Text'] = df['Text'].apply(self.preprocess_text)
        
        # Preparar features y target
        X = df['Processed_Text']
        y = self.prepare_target(df)
        
        # División del conjunto de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        ) # stratify asegura que la proporción de clases en el conjunto de entrenamiento y prueba sea la misma que en el dataset original, importante para problemas de clasificación desbalanceados
        
        # Entrenamiento
        self.pipeline.fit(X_train, y_train)
        
        # Evaluación con validación cruzada (10 pliegues para mayor robustez)
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
        
        # Predicciones finales
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
        """Predice si un texto contiene mensajes de odio"""
        if isinstance(text, str):
            text = [text]
        return self.pipeline.predict_proba(text)[:, 1]
    
    def save_model(self, path):
        """Guarda el modelo entrenado"""
        joblib.dump(self.pipeline, path)
    
    @classmethod
    def load_model(cls, path):
        """Carga un modelo guardado"""
        detector = cls()
        detector.pipeline = joblib.load(path)
        return detector

# Interfaz Streamlit
def create_streamlit_app():
    st.title("Advanced YouTube Hate Speech Detector (LSTM)")
    
    try:
        if os.path.exists('lstm_model.pth'):
            checkpoint = torch.load('lstm_model.pth')
            # Usar los parámetros originales para mantener compatibilidad
            model = ImprovedLSTMModel(1000, 128, 1)  # Volver a hidden_size=128
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
                            
                            # Display results with improved formatting
                            st.subheader("Analysis Results")
                            hate_comments = [r for r in results if r['is_hate']]
                            
                            # Metrics in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Comments", len(comments))
                            with col2:
                                st.metric("Hate Comments", len(hate_comments))
                            with col3:
                                st.metric("Hate Percentage", 
                                         f"{(len(hate_comments)/len(comments))*100:.1f}%")
                            
                            # Create tabs for different views
                            tab1, tab2 = st.tabs(["All Comments", "Hate Comments Only"])
                            
                            with tab1:
                                for result in results:
                                    with st.container():
                                        if result['is_hate']:
                                            st.error(
                                                f"⚠️ Probability: {result['probability']:.2%}\n"
                                                f"Comment: {result['comment']}"
                                            )
                                        else:
                                            st.success(
                                                f"✅ Probability: {result['probability']:.2%}\n"
                                                f"Comment: {result['comment']}"
                                            )
                            
                            with tab2:
                                if hate_comments:
                                    for result in hate_comments:
                                        st.error(
                                            f"⚠️ Probability: {result['probability']:.2%}\n"
                                            f"Comment: {result['comment']}"
                                        )
                                else:
                                    st.success("No hate comments found! 🎉")
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

if __name__ == "__main__":
    create_streamlit_app()