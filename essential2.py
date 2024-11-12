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
from sklearn.preprocessing import StandardScaler
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
def create_streamlit_app(detector, metrics):
    st.title("Detector de Mensajes de Odio")
    
    # Cargar modelo
    try:
        detector = HateSpeechDetector.load_model('hate_speech_model.pkl')
    except:
        st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
        return
    
    # Área de texto para input
    text_input = st.text_area("Introduce el texto a analizar:")
    
    if st.button("Analizar"):
        if text_input:
            probability = detector.predict(text_input)[0]
            st.write(f"Probabilidad de contenido de odio: {probability:.2%}")
            
            if probability > 0.5:
                st.error("⚠️ Este texto puede contener mensajes de odio.")
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
    
    # Guardar el modelo
    detector.save_model('hate_speech_model.pkl')
    
    # Ejecutar la app
    create_streamlit_app(detector, metrics)
