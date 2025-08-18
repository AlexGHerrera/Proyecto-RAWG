import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import time
import json
import os
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ------------------------------
# Configuración de la página
# ------------------------------
st.set_page_config(
    page_title="RAWG Games Intelligence",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .tech-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .content-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .example-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .highlight-text {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Configuración API y dataset
# ------------------------------
API_BASE_URL = "http://51.20.113.231"

def check_api_status():
    """Verificar el estado de la API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

@st.cache_data
def load_dataset():
    try:
        # Obtener ruta absoluta del dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "classification_dataset_v3.csv")
        df = pd.read_csv(dataset_path)
        return df
    except:
        return None

df = load_dataset()

# ------------------------------
# Funciones auxiliares
# ------------------------------
def api_post(endpoint, payload=None, timeout=30):
    try:
        resp = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except:
        return None

def api_get(endpoint, timeout=30):
    try:
        resp = requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except:
        return None

# ------------------------------
# Páginas
# ------------------------------
def show_portada():
    # Solo imagen de portada centrada
    try:
        # Obtener ruta absoluta de la imagen
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "portada.png")
        st.image(image_path, use_container_width=True)
    except:
        st.markdown("""
        <div class="main-header">
            <h1>RAWG Games Intelligence</h1>
            <p>Dashboard de Análisis y Predicción de Videojuegos</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>500K+</h3>
            <p>Juegos en BD</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>5</h3>
            <p>Endpoints API</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>SQLCoder-7B-2</h3>
            <p>Modelo NL→SQL</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>AWS</h3>
            <p>Infraestructura</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Descripción del proyecto
    st.subheader("Descripción")
    st.write("""
    **RAWG Games Intelligence** es un sistema completo que permite:
    
    - **Consultas en Lenguaje Natural**: Convierte preguntas en español e inglés a consultas SQL avanzadas
    - **Predicción de Éxito**: Estima el éxito de videojuegos usando características de diseño
    - **Análisis de Datos**: Visualizaciones automáticas con JOINs complejos y agregaciones
    - **Arquitectura Escalable**: Pipeline completo en AWS con PostgreSQL normalizado
    """)
    
    # Tecnologías utilizadas
    st.subheader("Stack Tecnológico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Backend & API</h4>
            <span class="tech-badge">FastAPI v5</span>
            <span class="tech-badge">PostgreSQL</span>
            <span class="tech-badge">SQLCoder-7B-2</span>
            <span class="tech-badge">Scikit-learn</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Infraestructura</h4>
            <span class="tech-badge">AWS EC2</span>
            <span class="tech-badge">AWS Lambda</span>
            <span class="tech-badge">AWS S3</span>
        </div>
        """, unsafe_allow_html=True)

def show_ml_process():
    st.markdown("""
    <div class="section-header">
        <h1>Proceso de Machine Learning</h1>
        <p>Análisis completo del proceso de EDA, Feature Engineering y entrenamiento de modelos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar proceso completo
    show_eda_process()
    show_feature_engineering()
    show_model_training_results()

def show_eda_process():
    """Mostrar el proceso de Análisis Exploratorio de Datos"""
    st.markdown("""
    <div class="content-card">
        <h2>1. Análisis Exploratorio de Datos (EDA)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h3>Problema del Dataset Desbalanceado</h3>
        <p><strong>Distribución Original:</strong></p>
        <ul>
            <li>High Success: 0.7% (Rating ≥4.0 + Added ≥1,000)</li>
            <li>Moderate Success: 8.0% (Rating ≥3.0 + Added ≥100)</li>
            <li>Low Success: 91.3% (El resto)</li>
            <li><strong>Ratio de desbalance: 127:1</strong> (inmanejable para ML)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico de distribución original vs optimizada
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribución Original (Problemática)**")
        original_data = pd.DataFrame({
            'Categoría': ['High Success', 'Moderate Success', 'Low Success'],
            'Porcentaje': [0.7, 8.0, 91.3]
        })
        st.bar_chart(original_data.set_index('Categoría'))
    
    with col2:
        st.markdown("**Distribución Optimizada v3**")
        optimized_data = pd.DataFrame({
            'Categoría': ['High Success', 'Moderate Success', 'Low Success'],
            'Porcentaje': [6.2, 8.3, 85.5]
        })
        st.bar_chart(optimized_data.set_index('Categoría'))
    
    st.markdown("""
    <div class="success-box">
        <h3>Redefinición de Criterios de Éxito</h3>
        <p><strong>Criterios Optimizados v3:</strong></p>
        <ul>
            <li><strong>High Success</strong>: Rating ≥ 3.5 AND Added ≥ 50 (6.2%)</li>
            <li><strong>Moderate Success</strong>: Rating ≥ 2.5 AND Added ≥ 10 (8.3%)</li>
            <li><strong>Low Success</strong>: El resto (85.5%)</li>
            <li><strong>Nuevo ratio: 1:14</strong> (manejable con class weights)</li>
        </ul>
        <p><strong>Justificación:</strong></p>
        <ul>
            <li>Basado en análisis de percentiles empíricos</li>
            <li>Mantiene validez conceptual del éxito</li>
            <li>Permite uso de técnicas estándar de ML</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_feature_engineering():
    """Mostrar el proceso de Feature Engineering"""
    st.markdown("""
    <div class="content-card">
        <h2>2. Feature Engineering</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>Evolución: Features Genéricas → Features Específicas</h3>
        <p><strong>Problema v2:</strong> 13 features genéricas con ceiling effect</p>
        <p><strong>Solución v3:</strong> 21 features específicas basadas en correlación empírica</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar estructura de features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Features Base (4):**
        - n_genres: Número de géneros
        - n_platforms: Número de plataformas
        - n_tags: Número de tags
        - release_year: Año de lanzamiento
        """)
        
        st.markdown("""
        **Features de Géneros Específicos (5):**
        - is_top_genre_1: Action
        - is_top_genre_2: Adventure
        - is_top_genre_3: Strategy
        - is_top_genre_4: RPG
        - is_top_genre_5: Shooter
        """)
    
    with col2:
        st.markdown("""
        **Features de Plataformas Específicas (5):**
        - is_top_platform_1: PC
        - is_top_platform_2: PlayStation
        - is_top_platform_3: Xbox
        - is_top_platform_4: Nintendo
        - is_top_platform_5: Mobile
        """)
        
        st.markdown("""
        **Features de Tags Específicos (5):**
        - is_top_tag_1: Multiplayer
        - is_top_tag_2: Singleplayer
        - is_top_tag_3: Atmospheric
        - is_top_tag_4: Story Rich
        - is_top_tag_5: Open World
        """)
    
    st.markdown("""
    **Features Derivadas (2):**
    - is_optimal_duration: Duración en rango óptimo
    - playtime: Horas de juego promedio
    
    ### Justificación de Features Específicas
    
    Las features específicas superan a las genéricas porque:
    - **Correlación empírica**: Basadas en análisis de datos reales
    - **Poder predictivo**: Capturan patrones específicos de éxito
    - **Interpretabilidad**: Cada feature tiene significado de negocio claro
    - **Escalabilidad**: Fácil agregar nuevas categorías específicas
    """)
    
    # Mostrar ejemplo de dataset
    if df is not None:
        st.subheader("Muestra del Dataset Final")
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Registros", f"{len(df):,}")
        with col2:
            st.metric("Features", "21")
        with col3:
            st.metric("Clases", "3")

def show_model_training_results():
    """Mostrar resultados del entrenamiento de modelos"""
    st.markdown("""
    <div class="content-card">
        <h2>3. Entrenamiento y Resultados de Modelos</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Estrategia para Dataset Desbalanceado
    
    **Modelos Seleccionados:**
    - **Logistic Regression**: Baseline interpretable con class weights
    - **Random Forest**: Robusto con features heterogéneas
    - **XGBoost**: Estado del arte en datos tabulares
    - **Neural Network**: Capacidad de abstracción optimizada
    
    **Técnicas de Balanceado:**
    - Class weights dinámicos calculados automáticamente
    - Métricas robustas (ROC-AUC, F1-Score macro)
    - Validación estratificada para mantener proporciones
    """)
    
    # Resultados de modelos
    st.subheader("Resultados Comparativos")
    
    results_data = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost', 'Neural Network', 'Logistic Regression'],
        'ROC-AUC Macro': [0.8789, 0.8651, 0.8283, 0.8007],
        'F1-Score Macro': [0.5726, 0.5481, 0.4838, 0.4937],
        'Accuracy': [0.8376, 0.7844, 0.7235, 0.7455]
    })
    
    st.dataframe(results_data.round(4))
    
    # Gráfico de comparación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC-AUC
    ax1.barh(results_data['Modelo'], results_data['ROC-AUC Macro'])
    ax1.set_xlabel('ROC-AUC Macro')
    ax1.set_title('Comparación ROC-AUC Macro')
    ax1.set_xlim(0.7, 0.9)
    
    # Accuracy
    ax2.barh(results_data['Modelo'], results_data['Accuracy'])
    ax2.set_xlabel('Accuracy')
    ax2.set_title('Comparación Accuracy')
    ax2.set_xlim(0.7, 0.9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    ### Análisis de Resultados
    
    **Mejor Modelo: Random Forest**
    - ROC-AUC Macro: 0.8789
    - Accuracy: 83.76%
    - **Objetivo alcanzado: Accuracy ≥ 80%**
    
    **Factores de Éxito:**
    - Features específicas vs genéricas
    - Balance natural manejable (1:14)
    - Class weights optimizados
    - Validación estratificada robusta
    
    **Métricas Clave:**
    - **ROC-AUC Macro**: Métrica principal para datasets desbalanceados
    - **F1-Score Macro**: Rendimiento equilibrado entre clases
    - **Accuracy**: Métrica secundaria con interpretación cuidadosa
    """)
    
    # Class weights utilizados
    st.subheader("Class Weights Optimizados")
    weights_data = pd.DataFrame({
        'Clase': ['High Success', 'Low Success', 'Moderate Success'],
        'Weight': [5.42, 0.39, 4.00],
        'Interpretación': ['Penaliza 5.4x los errores', 'Clase mayoritaria (base)', 'Penaliza 4.0x los errores']
    })
    st.dataframe(weights_data)
    
    st.markdown("""
    ### Conclusiones del Proceso ML
    
    **Lecciones Aprendidas:**
    1. **Feature Engineering específico** supera features genéricas
    2. **Balance natural** es preferible al artificial
    3. **Class weights** efectivos para ratios 1:14
    4. **Random Forest** robusto para features heterogéneas
    5. **Validación estratificada** crítica para evaluación
    
    **Aplicación en Producción:**
    - Modelo desplegado en API para predicción en tiempo real
    - Input: géneros, plataformas, tags, duración estimada
    - Output: probabilidad de éxito en 3 categorías
    """)

def show_example_analytics():
    """Mostrar gráfico único de ejemplo cuando no hay dataset local"""
    st.markdown("""
    <div class="content-card">
        <h3>⚙️ Configuración del Juego</h3>
        <p>Define las características de tu videojuego:</p>
    </div>
    """, unsafe_allow_html=True)
    genres_data = pd.DataFrame({
        'Género': ['Action', 'RPG', 'Strategy', 'Adventure', 'Shooter', 'Sports', 'Racing', 'Puzzle'],
        'Cantidad': [45000, 32000, 28000, 25000, 22000, 18000, 15000, 12000]
    })
    st.bar_chart(genres_data.set_index('Género'))

def show_nl_sql():
    st.markdown("""
    <div class="section-header">
        <h1>Consultas NL→SQL</h1>
        <p>Convierte preguntas en lenguaje natural a consultas SQL usando SQLCoder-7B-2</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplos de consultas
    st.markdown("""
    <div class="content-card">
        <h3>Ejemplos de Consultas</h3>
        <p>Consultas optimizadas para generar respuestas SQL:</p>
    </div>
    """, unsafe_allow_html=True)
    
    examples = [
        "Show me the top 5 highest rated games",
        "RPG games with rating above 4.0", 
        "Games available on PlayStation",
        "Count of games by genre",
        "Average rating by platform",
        "Games released between 2020 and 2023",
        "Muéstrame juegos de RPG con valoración superior a 4.0",
        "Juegos disponibles en PlayStation"
    ]
    
    selected_example = st.selectbox(
        "O selecciona un ejemplo:",
        ["Selecciona un ejemplo..."] + examples
    )
    
    # Input de consulta
    question = st.text_input(
        "Escribe tu pregunta (en inglés o español):",
        placeholder="Ejemplo: Show me RPG games with rating above 4.0 / Muéstrame juegos RPG con valoración superior a 4.0"
    )
    
    if selected_example != "Selecciona un ejemplo...":
        question = selected_example
    
    if st.button("Consultar", key="consultar_nl_sql"):
        if question:
            with st.spinner("Procesando consulta..."):
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ask-text",
                        json={"question": question},
                        timeout=30
                    )
                    
                    execution_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("success"):
                            st.success(f"Consulta procesada en {execution_time:.2f}s")
                            
                            # Mostrar SQL generado
                            st.subheader("SQL Generado:")
                            st.code(data["sql"], language="sql")
                            
                            # Mostrar resultados
                            if data["data"]:
                                st.subheader("Resultados:")
                                df_results = pd.DataFrame(data["data"])
                                st.dataframe(df_results)
                            else:
                                st.info("La consulta no devolvió resultados")
                        else:
                            st.error(f"Error: {data.get('error', 'Error desconocido')}")
                    else:
                        st.error(f"Error del servidor: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")
        else:
            st.warning("Por favor, ingresa una pregunta")

def show_visualizaciones():
    st.markdown("""
    <div class="section-header">
        <h1>Visualizaciones Interactivas</h1>
        <p>Genera visualizaciones interactivas usando el endpoint /ask-visual</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ejemplos de visualizaciones
    st.markdown("""
    <div class="content-card">
        <h3>Ejemplos de Visualizaciones</h3>
        <p>Consultas optimizadas para generar gráficos interactivos:</p>
    </div>
    """, unsafe_allow_html=True)
    
    viz_examples = [
        "count games by genre",
        "average rating by year", 
        "games by platform",
        "rating distribution",
        "top 10 highest rated games",
        "games released by year"
    ]
    
    selected_example = st.selectbox(
        "O selecciona un ejemplo:",
        ["Selecciona un ejemplo..."] + viz_examples
    )
    
    # Input de consulta
    question = st.text_input(
        "Escribe tu consulta para generar visualización (inglés o español):",
        placeholder="Ej: Count of games by genre / Cantidad de juegos por género"
    )
    
    if selected_example != "Selecciona un ejemplo...":
        question = selected_example
    
    if st.button("Generar Visualización", key="generar_viz"):
        if question:
            with st.spinner("Generando visualización..."):
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ask-visual",
                        json={"question": question},
                        timeout=30
                    )
                    
                    execution_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("success"):
                            st.success(f"Visualización generada en {execution_time:.2f}s")
                            
                            # Mostrar SQL generado
                            st.subheader("SQL Generado:")
                            st.code(data["sql"], language="sql")
                            
                            # Mostrar visualización
                            if data.get("visualization"):
                                st.subheader("Visualización:")
                                st.plotly_chart(data["visualization"], use_container_width=True)
                            
                            # Mostrar datos
                            if data["data"]:
                                st.subheader("Datos:")
                                df_results = pd.DataFrame(data["data"])
                                st.dataframe(df_results)
                            else:
                                st.info("La consulta no devolvió datos")
                        else:
                            st.error(f"Error: {data.get('error', 'Error desconocido')}")
                    else:
                        st.error(f"Error del servidor: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")
        else:
            st.warning("Por favor, ingresa una consulta")

def show_prediccion_exito():
    st.markdown("""
    <div class="section-header">
        <h1>Predicción de Éxito</h1>
        <p>Predice el éxito de un videojuego usando el modelo v3 entrenado</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Presets de ejemplo
    st.markdown("""
    <div class="content-card">
        <h3>Ejemplos por Categoría</h3>
        <p>Selecciona un preset de ejemplo para cada categoría de éxito:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("High Success", key="preset_high"):
            st.session_state.preset_genres = ["Action", "Adventure", "Strategy"]
            st.session_state.preset_platforms = ["PC", "PlayStation 4", "Xbox One", "Nintendo Switch", "PlayStation 5"]
            st.session_state.preset_tags = "Singleplayer, Multiplayer, Atmospheric, Story Rich, Open World"
            st.session_state.preset_hours = 15
            st.session_state.preset_year = 2025
    
    with col2:
        if st.button("Moderate Success", key="preset_moderate"):
            st.session_state.preset_genres = ["Action", "Adventure"]
            st.session_state.preset_platforms = ["PC", "PlayStation 4", "Xbox One"]
            st.session_state.preset_tags = "Singleplayer, Multiplayer"
            st.session_state.preset_hours = 5
            st.session_state.preset_year = 2025
    
    with col3:
        if st.button("Low Success", key="preset_low"):
            st.session_state.preset_genres = ["Action"]
            st.session_state.preset_platforms = ["PC"]
            st.session_state.preset_tags = "Singleplayer"
            st.session_state.preset_hours = 2
            st.session_state.preset_year = 2025
    
    # Formulario de predicción
    st.markdown("""
    <div class="content-card">
        <h3>Configuración del Juego</h3>
        <p>Define las características de tu videojuego:</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            genres = st.multiselect(
                "Géneros:",
                ["Action", "RPG", "Strategy", "Adventure", "Shooter", "Simulation", 
                 "Sports", "Racing", "Puzzle", "Fighting", "Platform"],
                default=st.session_state.get("preset_genres", ["Action"])
            )
            
            platforms = st.multiselect(
                "Plataformas:",
                ["PC", "PlayStation 5", "PlayStation 4", "Xbox Series S/X", 
                 "Xbox One", "Nintendo Switch", "Mobile"],
                default=st.session_state.get("preset_platforms", ["PC"])
            )
        
        with col2:
            tags = st.text_input(
                "Tags (separados por comas):",
                value=st.session_state.get("preset_tags", "multiplayer, aventura")
            )
            
            estimated_hours = st.slider(
                "Horas estimadas de juego:",
                min_value=1, max_value=200, 
                value=st.session_state.get("preset_hours", 30)
            )
            
            preset_year = st.session_state.get("preset_year", 2024)
            year_options = list(range(2024, 2027))
            
            # Validar que el año preset esté en las opciones
            if preset_year in year_options:
                year_index = preset_year - 2024
            else:
                year_index = 0
            
            planned_year = st.selectbox(
                "Año planeado:",
                year_options,
                index=year_index
            )
        
        submitted = st.form_submit_button("Predecir Éxito")
        
        if submitted:
            # Preparar datos para la API (formato correcto según predict.py)
            prediction_data = {
                "genres": genres,  # Lista directa
                "platforms": platforms,  # Lista directa
                "tags": tags.split(", ") if tags else [],  # Convertir string a lista
                "estimated_hours": float(estimated_hours),  # Float directo
                "release_year": int(planned_year)  # Int directo
            }
            
            with st.spinner("Calculando predicción..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/predict",  # Endpoint correcto
                        json=prediction_data,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Obtener datos del modelo
                        predicted_class = result.get("predicted_class", "low_success")
                        confidence = result.get("confidence", 0)
                        probabilities = result.get("probabilities", {})
                        
                        # Mapear clase a categoría y score
                        class_mapping = {
                            "high_success": {"category": "Alto", "score": probabilities.get("high_success", 0)},
                            "moderate_success": {"category": "Medio", "score": probabilities.get("moderate_success", 0)},
                            "low_success": {"category": "Bajo", "score": probabilities.get("low_success", 0)}
                        }
                        
                        current_class = class_mapping.get(predicted_class, class_mapping["low_success"])
                        category = current_class["category"]
                        class_probability = current_class["score"]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Clase Predicha", predicted_class.replace("_", " ").title())
                        
                        with col2:
                            st.metric("Categoría", category)
                        
                        with col3:
                            st.metric("Confianza", f"{confidence:.1%}")
                        
                        # Visualización de la probabilidad de la clase predicha
                        st.progress(class_probability)
                        
                        # Mostrar todas las probabilidades
                        st.subheader("Probabilidades por Clase:")
                        for class_name, prob in probabilities.items():
                            display_name = class_name.replace("_", " ").title()
                            st.write(f"**{display_name}**: {prob:.1%}")
                        
                        # Interpretación basada en la clase predicha
                        if predicted_class == "high_success":
                            st.success("Alto potencial de éxito")
                        elif predicted_class == "moderate_success":
                            st.warning("Potencial moderado")
                        else:
                            st.error("Bajo potencial de éxito")
                        
                        # Recomendaciones
                        if "recommendations" in result:
                            st.subheader("Recomendaciones:")
                            for rec in result["recommendations"]:
                                st.write(f"• {rec}")
                    
                    else:
                        st.error(f"Error en la predicción: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")

def show_arquitectura():
    st.header("Arquitectura e Infraestructura")
    
    # Diagrama de arquitectura
    st.subheader("Diagrama de Arquitectura")
    try:
        st.image("../docs/arquitectura_aws.png", caption="Arquitectura AWS del Sistema")
    except:
        pass
    
    # Pipeline de datos detallado
    st.subheader("Pipeline de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>1. Ingesta</h4>
            <p><strong>AWS Lambda + EventBridge</strong></p>
            <ul>
                <li>Extracción diaria desde RAWG API</li>
                <li>Almacenamiento en S3</li>
                <li>Procesamiento incremental</li>
                <li>Manejo de rate limits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>2. Procesamiento</h4>
            <p><strong>AWS Lambda Loader</strong></p>
            <ul>
                <li>Transformación y limpieza</li>
                <li>Validación de integridad</li>
                <li>Carga a PostgreSQL RDS</li>
                <li>Logs de auditoría</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>3. Servicio</h4>
            <p><strong>FastAPI en EC2</strong></p>
            <ul>
                <li>API NL→SQL con SQLCoder-7B-2</li>
                <li>Predicción ML</li>
                <li>Visualizaciones automáticas</li>
                <li>Documentación Swagger</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Infraestructura detallada
    st.subheader("Infraestructura AWS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Servicios AWS</h4>
            <span class="tech-badge">EC2 t3.medium</span>
            <span class="tech-badge">RDS PostgreSQL</span>
            <span class="tech-badge">S3 Bucket</span>
            <span class="tech-badge">Lambda Functions</span>
            <span class="tech-badge">EventBridge</span>
            <span class="tech-badge">CloudWatch</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>Base de Datos</h4>
            <p><strong>PostgreSQL 13+ en RDS</strong></p>
            <ul>
                <li>500K+ registros de juegos</li>
                <li>Vista optimizada games_complete</li>
                <li>Índices para consultas rápidas</li>
                <li>Backups automáticos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Despliegue EC2</h4>
            <span class="tech-badge">Nginx</span>
            <span class="tech-badge">Systemd</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>Modelos ML</h4>
            <p><strong>Entrenamiento y Producción</strong></p>
            <ul>
                <li>SQLCoder-7B-2 para NL→SQL</li>
                <li>Random Forest para predicción</li>
                <li>Features de diseño optimizadas</li>
                <li>Validación cruzada</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas de rendimiento
    st.subheader("Métricas de Rendimiento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Latencia API", "< 2s", "Promedio")
    with col2:
        st.metric("Disponibilidad", "99.5%", "Uptime")
    with col3:
        st.metric("Consultas/día", "~1000", "Capacidad")
    with col4:
        st.metric("Precisión SQL", "97.5%", "SQLCoder-7B-2")


# ------------------------------
# Menú lateral
# ------------------------------
st.sidebar.title("Navegación")

# Estado de la API
api_online = check_api_status()
status_class = "status-online" if api_online else "status-offline"
status_text = "API Online" if api_online else "API Offline"

st.sidebar.markdown(f"""
<div class="api-status {status_class}">
    {status_text}
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Selecciona una página:",
    ["Inicio", "Consultas NL→SQL", "Visualizaciones", "Predicción de Éxito", 
     "Proceso ML", "Arquitectura"]
)

# Routing de páginas
if page == "Inicio":
    show_portada()
elif page == "Consultas NL→SQL":
    show_nl_sql()
elif page == "Visualizaciones":
    show_visualizaciones()
elif page == "Predicción de Éxito":
    show_prediccion_exito()
elif page == "Proceso ML":
    show_ml_process()
elif page == "Arquitectura":
    show_arquitectura()
    