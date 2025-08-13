"""
predict.py
==========

Este módulo proporciona la implementación real del modelo de predicción v3
para predecir el éxito de videojuegos basándose en sus características de diseño.
Utiliza un Random Forest entrenado que clasifica juegos en tres categorías:
high_success, moderate_success, y low_success.
"""

from __future__ import annotations

from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json

# Variables globales para el modelo cargado
_model = None
_label_encoder = None
_metadata = None

# Top categorías basadas en el entrenamiento del modelo v3
TOP_GENRES = ['Action', 'Adventure', 'RPG', 'Shooter', 'Strategy']
TOP_PLATFORMS = ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch', 'PlayStation 5']
TOP_TAGS = ['Singleplayer', 'Multiplayer', 'Atmospheric', 'Story Rich', 'Open World']

def load_model_v3():
    """Carga el modelo v3 y sus componentes."""
    global _model, _label_encoder, _metadata
    
    if _model is not None:
        return  # Ya está cargado
    
    # Buscar archivos del modelo v3
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "data" / "model_v3"
    
    # Encontrar los archivos más recientes
    model_files = list(model_dir.glob("best_random_forest_v3_*.pkl"))
    encoder_files = list(model_dir.glob("label_encoder_v3_*.pkl"))
    metadata_files = list(model_dir.glob("model_metadata_v3_*.json"))
    
    if not model_files or not encoder_files:
        raise FileNotFoundError("No se encontraron archivos del modelo v3")
    
    # Usar el archivo más reciente
    model_file = sorted(model_files)[-1]
    encoder_file = sorted(encoder_files)[-1]
    
    # Cargar componentes
    _model = joblib.load(model_file)
    _label_encoder = joblib.load(encoder_file)
    
    # Cargar metadata si existe, sino usar valores por defecto
    if metadata_files:
        metadata_file = sorted(metadata_files)[-1]
        with open(metadata_file, 'r') as f:
            _metadata = json.load(f)
    else:
        # Metadata por defecto como fallback
        _metadata = {
            "model_name": "Random Forest",
            "version": "v3",
            "features": ["n_genres", "n_platforms", "n_tags", "release_year", "playtime"],
            "target_classes": ["high_success", "low_success", "moderate_success"]
        }

def preprocess_features(input_data: Dict[str, Any]) -> np.ndarray:
    """Preprocesa las características de entrada para el modelo v3."""
    
    # Extraer datos básicos
    genres = input_data.get('genres', [])
    platforms = input_data.get('platforms', [])
    tags = input_data.get('tags', [])
    estimated_hours = input_data.get('estimated_hours', 0.0)
    release_year = input_data.get('release_year', 2024)
    
    # Crear features básicas
    features = {
        'n_genres': len(genres),
        'n_platforms': len(platforms), 
        'n_tags': len(tags),
        'release_year': release_year,
        'playtime': estimated_hours
    }
    
    # Features de top géneros
    for i, genre in enumerate(TOP_GENRES, 1):
        features[f'is_top_genre_{i}'] = 1 if genre in genres else 0
    
    # Features de top plataformas
    for i, platform in enumerate(TOP_PLATFORMS, 1):
        features[f'is_top_platform_{i}'] = 1 if platform in platforms else 0
    
    # Features de top tags
    for i, tag in enumerate(TOP_TAGS, 1):
        features[f'is_top_tag_{i}'] = 1 if tag in tags else 0
    
    # Feature de duración óptima (basado en análisis previo)
    features['is_optimal_duration'] = 1 if 10 <= estimated_hours <= 50 else 0
    
    # Convertir a array en el orden correcto según metadatos
    feature_order = _metadata['features']
    feature_array = np.array([features.get(feat, 0) for feat in feature_order])
    
    return feature_array.reshape(1, -1)

def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza predicción de éxito de videojuego usando el modelo v3.

    Parámetros
    ----------
    input_data : dict
        Diccionario con características del juego:
        - genres: Lista de géneros
        - platforms: Lista de plataformas
        - tags: Lista de tags
        - estimated_hours: Horas estimadas de juego
        - release_year: Año de lanzamiento

    Retorna
    -------
    dict
        Diccionario con predicted_class, confidence y probabilities
    """
    
    # Cargar modelo si no está cargado
    if _model is None:
        load_model_v3()
    
    # Preprocesar características
    features = preprocess_features(input_data)
    
    # Realizar predicción
    prediction = _model.predict(features)[0]
    probabilities = _model.predict_proba(features)[0]
    
    # Decodificar clase predicha
    predicted_class = _label_encoder.inverse_transform([prediction])[0]
    
    # Calcular confianza (probabilidad máxima)
    confidence = float(np.max(probabilities))
    
    # Crear diccionario de probabilidades por clase
    class_names = _metadata['target_classes']
    prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict
    }