import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import streamlit as st
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import randint, uniform

class HateSpeechDetector:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
            ('scaler', StandardScaler(with_mean=False)),  # Normalizar características
            ('classifier', RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
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
        
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train.to_frame(), y_train)
        
        param_dist = {
            'vectorizer__max_features': randint(2000, 5000),
            'classifier__n_estimators': randint(100, 500),
            'classifier__max_depth': randint(10, 50),
            'classifier__min_samples_split': randint(2, 20),
            'classifier__min_samples_leaf': randint(1, 10),
            'classifier__max_features': ['sqrt', 'log2']
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(
            self.pipeline, param_distributions=param_dist, 
            n_iter=50, cv=cv, scoring='f1_macro', n_jobs=-1, random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train_resampled['Text'], y_train_resampled)
        
        self.pipeline = random_search.best_estimator_
        
        train_pred = self.pipeline.predict(X_train_resampled['Text'])
        test_pred = self.pipeline.predict(X_test)
        
        train_acc = np.mean(train_pred == y_train_resampled)
        test_acc = np.mean(test_pred == y_test)
        
        train_f1 = f1_score(y_train_resampled, train_pred)
        test_f1 = f1_score(y_test, test_pred)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'overfitting': train_acc - test_acc,
            'classification_report': classification_report(y_test, test_pred)
        }
    
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

def create_streamlit_app(detector, metrics):
    st.title("Detector de Mensajes de Odio")
    
    text_input = st.text_area("Introduce el texto a analizar:")
    
    if st.button("Analizar"):
        if text_input:
            probability = detector.predict([text_input])[0]
            st.write(f"Probabilidad de contenido de odio: {probability:.2%}")
            
            if probability > 0.5:
                st.error("⚠️ Este texto puede contener mensajes de odio.")
            else:
                st.success("✅ Este texto parece seguro.")
            
            st.subheader("Métricas del modelo:")
            metrics_df = pd.DataFrame({
                'Métrica': ['Precisión de entrenamiento', 'Precisión de prueba', 'F1 de entrenamiento', 'F1 de prueba', 'Overfitting'],
                'Valor': [
                    f"{metrics['train_accuracy']:.4f}",
                    f"{metrics['test_accuracy']:.4f}",
                    f"{metrics['train_f1']:.4f}",
                    f"{metrics['test_f1']:.4f}",
                    f"{metrics['overfitting']:.4f}"
                ]
            })
            st.table(metrics_df)
        else:
            st.warning("Por favor, introduce algún texto para analizar.")

if __name__ == "__main__":
    df = pd.read_csv('youtoxic_english_1000.csv')
    detector = HateSpeechDetector()
    metrics = detector.train(df)
    print("Métricas del modelo:")
    for key, value in metrics.items():
        if key != 'classification_report':
            print(f"{key}: {value}")
    print(f"Classification Report:\n{metrics['classification_report']}")
    
    detector.save_model('hate_speech_model_improved.pkl')
    
    create_streamlit_app(detector, metrics)