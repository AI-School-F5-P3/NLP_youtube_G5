import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import streamlit as st
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

class HateSpeechDetector:
    def __init__(self):
        # Usar pipeline para estandarizar el proceso
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,  # Reducido para evitar overfitting
                ngram_range=(1, 2),
                min_df=2,  # Ignorar términos que aparecen en menos de 2 documentos
                max_df=0.95,  # Ignorar términos que aparecen en más del 95% de los documentos
                stop_words='english'
            )),
            ('scaler', StandardScaler(with_mean=False)),  # Normalizar características
            ('classifier', LogisticRegression(
                C=0.1,  # Aumentar regularización
                class_weight='balanced',  # Manejar desbalance de clases
                random_state=42,
                max_iter=1000
            ))
        ])
        
    def prepare_target(self, df):
        """Combina todas las columnas objetivo en una sola etiqueta de odio"""
        hate_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 
                       'IsObscene', 'IsHatespeech', 'IsRacist', 'IsNationalist', 
                       'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
        return (df[hate_columns].sum(axis=1) > 0).astype(int)
    
    def train(self, df):
        """Entrena el modelo con los datos proporcionados"""
        # Preparar features y target
        X = df['Text']
        y = self.prepare_target(df)
        
        # División del conjunto de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenamiento
        self.pipeline.fit(X_train, y_train)
        
        # Evaluación con validación cruzada
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
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    @classmethod
    def load_model(cls, path):
        """Carga un modelo guardado"""
        detector = cls()
        with open(path, 'rb') as f:
            detector.pipeline = pickle.load(f)
        return detector

# Interfaz Streamlit
def create_streamlit_app():
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

if __name__ == "__main__":
    # Cargar y entrenar el modelo
    df = pd.read_csv('youtoxic_english_1000.csv')
    detector = HateSpeechDetector()
    metrics = detector.train(df)
    print("Métricas del modelo:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Guardar el modelo
    detector.save_model('hate_speech_model.pkl')
    
    # Ejecutar la app
    create_streamlit_app()