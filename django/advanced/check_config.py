# check_config.py
import joblib
import os

def check_model_config():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    config_path = os.path.join(models_dir, 'model_config.pkl')
    
    try:
        config = joblib.load(config_path)
        print("Contenido de model_config.pkl:")
        for key, value in config.items():
            print(f"{key}: {value}")
            
        # Verificar que todos los parámetros necesarios están presentes
        required_params = [
            'vocab_size', 
            'embedding_dim', 
            'hidden_dim', 
            'n_layers', 
            'dropout', 
            'bidirectional',
            'metrics'
        ]
        
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            print("\nFaltan los siguientes parámetros:")
            print(missing_params)
        else:
            print("\nTodos los parámetros requeridos están presentes.")
            
    except Exception as e:
        print(f"Error al cargar la configuración: {e}")

if __name__ == "__main__":
    check_model_config()