# %% [markdown]
# # Batería de Pruebas - API RAWG v2
# 
# Este notebook prueba los 3 endpoints principales de la API RAWG v2:
# - `/predict` - Predicción de éxito de videojuegos
# - `/ask-text` - Consultas en lenguaje natural → SQL
# - `/ask-visual` - Consultas con visualización automática
# 
# **Requisitos**: La API debe estar ejecutándose en `http://localhost:8000`

# %%
import requests
import json
import time
from typing import Dict, Any
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Configuración de la API
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def print_section(title: str):
    """Imprime una sección con formato"""
    print(f"\n{'='*60}")
    print(f"[TEST] {title}")
    print('='*60)

def test_endpoint(endpoint: str, data: Dict[str, Any], description: str):
    """Prueba un endpoint y muestra los resultados"""
    print(f"\nPrueba: {description}")
    print(f"Endpoint: {endpoint}")
    print(f"Datos enviados: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}{endpoint}", 
                               json=data, 
                               headers=HEADERS,
                               timeout=30)
        duration = time.time() - start_time
        
        print(f"Tiempo de respuesta: {duration:.2f}s")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            if 'image/png' in response.headers.get('content-type', ''):
                print("Respuesta: Imagen PNG generada correctamente")
                return response.content
            else:
                result = response.json()
                print(f"Respuesta exitosa:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return result
        else:
            print(f"Error: {response.status_code}")
            print(f"Detalle: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None

# %% [markdown]
# ## 1. Pruebas del Endpoint `/predict`
# 
# Prueba la predicción de éxito de videojuegos basada en características de diseño.

# %%
print_section("PRUEBAS DEL ENDPOINT /predict")

# Prueba 1: Juego AAA típico
test_endpoint("/predict", {
    "genres": ["Action", "Adventure"],
    "platforms": ["PC", "PlayStation 5", "Xbox Series X"],
    "tags": ["Singleplayer", "Story Rich", "Open World"],
    "esrb_rating": "M",
    "estimated_hours": 45,
    "planned_year": 2024
}, "Juego AAA de acción/aventura")

# Prueba 2: Juego indie
test_endpoint("/predict", {
    "genres": ["Indie", "Puzzle"],
    "platforms": ["PC", "Nintendo Switch"],
    "tags": ["Casual", "Relaxing", "Minimalist"],
    "esrb_rating": "E",
    "estimated_hours": 8,
    "planned_year": 2024
}, "Juego indie de puzzles")

# Prueba 3: Juego multijugador competitivo
test_endpoint("/predict", {
    "genres": ["Action", "Shooter"],
    "platforms": ["PC", "PlayStation 5", "Xbox Series X"],
    "tags": ["Multiplayer", "Competitive", "FPS"],
    "esrb_rating": "M",
    "estimated_hours": 100,
    "planned_year": 2024
}, "Shooter multijugador competitivo")

# %% [markdown]
# ## 2. Pruebas del Endpoint `/ask-text`
# 
# Prueba la conversión de preguntas en lenguaje natural a consultas SQL.

# %%
print_section("PRUEBAS DEL ENDPOINT /ask-text")

# Consultas básicas
text_queries = [
    "¿Cuáles son los 10 juegos mejor valorados?",
    "¿Cuántos juegos hay por género?",
    "¿Cuál es el promedio de rating por género?",
    "¿Cuáles son los mejores juegos de RPG?",
    "¿Cuántos juegos hay en PC?",
    "¿Cuáles son los juegos más populares de 2023?",
    "¿Qué géneros tienen más juegos?",
    "¿Cuáles son los mejores juegos de acción?",
    "¿Cuántos juegos hay por plataforma?",
    "¿Cuáles son los juegos con mejor rating en PlayStation?"
]

for i, query in enumerate(text_queries, 1):
    test_endpoint("/ask-text", {
        "question": query
    }, f"Consulta {i}: {query}")
    
    # Pausa pequeña entre consultas
    time.sleep(0.5)

# %% [markdown]
# ## 3. Pruebas del Endpoint `/ask-visual`
# 
# Prueba la generación automática de visualizaciones basadas en consultas en lenguaje natural.

# %%
print_section("PRUEBAS DEL ENDPOINT /ask-visual")

# Consultas que generan diferentes tipos de gráficos
visual_queries = [
    "Distribución de juegos por género",
    "Top 10 géneros más populares", 
    "Promedio de rating por género",
    "Número de juegos por plataforma",
    "Mejores juegos de RPG",
    "Distribución de ratings",
    "Juegos lanzados por año",
    "Comparación de géneros por popularidad"
]

images_generated = []

for i, query in enumerate(visual_queries, 1):
    print(f"\nGenerando visualización {i}/{len(visual_queries)}")
    
    image_data = test_endpoint("/ask-visual", {
        "question": query
    }, f"Visualización: {query}")
    
    if image_data and isinstance(image_data, bytes):
        try:
            # Guardar imagen para revisión
            filename = f"test_visualization_{i}.png"
            with open(filename, 'wb') as f:
                f.write(image_data)
            print(f"Imagen guardada como: {filename}")
            images_generated.append((query, filename))
            
        except Exception as e:
            print(f"Error guardando imagen: {e}")
    
    # Pausa entre visualizaciones
    time.sleep(1)

# %% [markdown]
# ## 4. Resumen de Resultados
# 
# Resumen final de todas las pruebas realizadas.

# %%
print_section("RESUMEN DE RESULTADOS")

print("ENDPOINTS PROBADOS:")
print("- /predict - Predicción de éxito de videojuegos")
print("- /ask-text - Consultas texto → SQL") 
print("- /ask-visual - Consultas → Visualizaciones")

print(f"\nESTADÍSTICAS:")
print(f"Consultas de texto probadas: {len(text_queries)}")
print(f"Visualizaciones generadas: {len(images_generated)}")

if images_generated:
    print(f"\nIMÁGENES GENERADAS:")
    for query, filename in images_generated:
        print(f"   - {filename}: {query}")

print(f"\nPRUEBAS COMPLETADAS")
print("Revisa los resultados arriba para verificar el funcionamiento")
print("Las imágenes generadas están guardadas en el directorio actual")

# %% [markdown]
# ## 5. Pruebas de Casos Límite
# 
# Pruebas adicionales para verificar el manejo de errores y casos especiales.

# %%
print_section("PRUEBAS DE CASOS LÍMITE")

# Prueba con datos inválidos para /predict
test_endpoint("/predict", {
    "genres": [],  # Lista vacía
    "platforms": ["PC"],
    "tags": ["Test"],
    "esrb_rating": "Invalid",  # Rating inválido
    "estimated_hours": -5,  # Horas negativas
    "planned_year": 1990  # Año muy antiguo
}, "Datos inválidos para predicción")

# Prueba con pregunta vacía para /ask-text
test_endpoint("/ask-text", {
    "question": ""
}, "Pregunta vacía")

# Prueba con pregunta muy compleja para /ask-text
test_endpoint("/ask-text", {
    "question": "Dame una consulta súper compleja que involucre múltiples joins, subqueries, y funciones de ventana para analizar la correlación entre géneros, plataformas, y ratings considerando tendencias temporales"
}, "Pregunta muy compleja")

# Prueba con pregunta no relacionada con videojuegos
test_endpoint("/ask-text", {
    "question": "¿Cuál es la capital de Francia?"
}, "Pregunta no relacionada con videojuegos")

print(f"\nTODAS LAS PRUEBAS COMPLETADAS")
print("Revisa los resultados para evaluar la robustez de la API")

# %% [markdown]
# ## Notas de Uso
# 
# ### Para ejecutar este notebook:
# 
# 1. **Iniciar la API**: 
#    ```bash
#    cd api/api_v2
#    uvicorn main:app --reload --port 8000
#    ```
# 
# 2. **Ejecutar las pruebas**:
#    - Convertir a notebook: `jupytext --to notebook test_api_endpoints.py`
#    - O ejecutar directamente: `python test_api_endpoints.py`
# 
# 3. **Verificar resultados**:
#    - Consultas SQL generadas correctamente
#    - Imágenes PNG creadas para visualizaciones
#    - Predicciones numéricas razonables
# 
# ### Endpoints probados:
# - **POST /predict**: Predicción de éxito basada en features de diseño
# - **POST /ask-text**: Conversión de lenguaje natural a SQL
# - **POST /ask-visual**: Generación automática de gráficos
# 
# ### Casos de prueba incluidos:
# - Casos típicos y exitosos
# - Diferentes tipos de consultas
# - Variedad de visualizaciones
# - Casos límite y manejo de errores
