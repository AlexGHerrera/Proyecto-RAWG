import pickle
import json
import numpy as np
import sklearn
print(f"🔍 API usando scikit-learn version: {sklearn.__version__}")

# Cargar modelo v3
def load_model():
    model_dir = "../data/model_v3"
    
    # Cargar modelo Random Forest (versión actual)
    with open(f"{model_dir}/best_random_forest_v3_20250807_224107.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Cargar label encoder (versión actual)
    with open(f"{model_dir}/label_encoder_v3_20250807_224107.pkl", 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

def create_features(input_data):
    """Crea vector de 21 features para el modelo v3."""
    
    # Mapeos basados en EDA v3
    top_genres = ["Action", "Indie", "Adventure", "RPG", "Strategy"]
    top_platforms = ["PC", "PlayStation 4", "Xbox One", "Nintendo Switch", "PlayStation 5"]
    top_tags = ["Singleplayer", "Multiplayer", "Atmospheric", "Story Rich", "Open World"]
    
    features = np.zeros(21)
    
    # Features básicas
    features[0] = len(input_data.get('genres', []))
    features[1] = len(input_data.get('platforms', []))
    features[2] = len(input_data.get('tags', []))
    features[3] = input_data.get('release_year', 2024)
    
    # Géneros específicos
    user_genres = input_data.get('genres', [])
    for i, genre in enumerate(top_genres):
        features[4 + i] = 1 if genre in user_genres else 0
    
    # Plataformas específicas
    user_platforms = input_data.get('platforms', [])
    for i, platform in enumerate(top_platforms):
        features[9 + i] = 1 if platform in user_platforms else 0
    
    # Tags específicos
    user_tags = input_data.get('tags', [])
    for i, tag in enumerate(top_tags):
        features[14 + i] = 1 if tag in user_tags else 0
    
    # Features derivadas
    estimated_hours = input_data.get('estimated_hours', 0)
    features[19] = 1 if 10 <= estimated_hours <= 50 else 0
    features[20] = estimated_hours
    
    return features.reshape(1, -1)

def predict(input_data):
    """Predice éxito del videojuego."""
    model, label_encoder = load_model()
    
    # Crear features
    features = create_features(input_data)
    
    # Debug: mostrar features
    print(f"🔍 Features creadas: {features.flatten()}")
    
    # Predicción
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Debug: mostrar predicción raw
    print(f"🔍 Predicción raw: {prediction}")
    print(f"🔍 Probabilidades raw: {probabilities}")
    print(f"🔍 Clases del encoder: {label_encoder.classes_}")
    
    # Decodificar resultado
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))
    
    # Crear diccionario de probabilidades usando las clases correctas
    prob_dict = {}
    for i, class_name in enumerate(label_encoder.classes_):
        prob_dict[class_name] = float(probabilities[i])
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "debug_info": {
            "features_vector": features.flatten().tolist(),
            "raw_prediction": int(prediction),
            "raw_probabilities": probabilities.tolist(),
            "encoder_classes": label_encoder.classes_.tolist()
        }
    }