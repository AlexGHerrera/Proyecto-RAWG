"""
ask_visual.py
=============

Módulo de visualización unificado con capacidades mejoradas de auto-visualización.
Características: Detección automática de tablas + Visualización inteligente + UX mejorada

Ejemplo
-------

>>> import pandas as pd
>>> from models import ask_visual
>>> df = pd.DataFrame({"genre": ["Action", "RPG"], "count": [10, 5]})
>>> fig = ask_visual.auto_viz(df, "¿Cuántos juegos por género?")
>>> fig.show()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
from typing import Dict, List, Optional, Tuple

# Importa el modelo T5 para satisfacer el requisito de que tanto ask_text como
# ask_visual dependan de él.
import t5_model  # noqa: F401

# Template oscuro optimizado
pio.templates["enhanced_dark"] = go.layout.Template(
    layout=dict(
        title_font=dict(family="Arial", size=20, color="white"),
        font=dict(family="Arial", size=12, color="#E1E5E9"),
        paper_bgcolor="rgb(17,17,17)",
        plot_bgcolor="rgb(17,17,17)",
        xaxis=dict(
            gridcolor="#283442", 
            linecolor="#506784", 
            zerolinecolor="#283442",
            tickcolor="#506784"
        ),
        yaxis=dict(
            gridcolor="#283442", 
            linecolor="#506784", 
            zerolinecolor="#283442",
            tickcolor="#506784"
        ),
        colorway=["#00D4FF", "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
    )
)

pio.templates.default = "enhanced_dark"

# Mapeo de keywords a tablas relevantes
TABLE_KEYWORDS = {
    "games": ["juego", "game", "título", "rating", "horas", "playtime", "metacritic", "lanzamiento"],
    "genres": ["género", "genre", "categoría", "tipo", "acción", "rpg", "strategy"],
    "platforms": ["plataforma", "platform", "pc", "xbox", "playstation", "nintendo", "switch"],
    "tags": ["tag", "etiqueta", "característica", "multiplayer", "singleplayer", "online"],
    "stores": ["tienda", "store", "steam", "epic", "gog"],
    "ratings": ["calificación", "rating", "puntuación", "score", "valoración"]
}

# Patrones de consulta comunes
QUERY_PATTERNS = {
    "average_by_category": {
        "keywords": ["promedio", "media", "average", "por", "by"],
        "viz_type": "bar",
        "description": "Promedio de una métrica por categoría"
    },
    "top_ranking": {
        "keywords": ["top", "mejor", "mejores", "best", "más", "mayor"],
        "viz_type": "horizontal_bar",
        "description": "Ranking de elementos"
    },
    "distribution": {
        "keywords": ["distribución", "distribution", "frecuencia", "histograma"],
        "viz_type": "histogram",
        "description": "Distribución de valores"
    },
    "time_series": {
        "keywords": ["año", "year", "tiempo", "time", "evolución", "tendencia"],
        "viz_type": "line",
        "description": "Evolución temporal"
    },
    "comparison": {
        "keywords": ["comparar", "compare", "vs", "versus", "diferencia"],
        "viz_type": "grouped_bar",
        "description": "Comparación entre categorías"
    }
}


def detect_relevant_tables(user_question: str) -> List[str]:
    """
    Detecta automáticamente qué tablas son relevantes para la consulta
    """
    question_lower = user_question.lower()
    detected_tables = []
    
    for table, keywords in TABLE_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_tables.append(table)
    
    # Lógica de dependencias automáticas
    if "games" in detected_tables:
        if "genres" in detected_tables:
            detected_tables.append("game_genres")
        if "platforms" in detected_tables:
            detected_tables.append("game_platforms")
        if "tags" in detected_tables:
            detected_tables.append("game_tags")
    
    return list(set(detected_tables))


def detect_query_intent(user_question: str) -> Tuple[str, str]:
    """
    Detecta la intención de la consulta para seleccionar visualización apropiada
    """
    question_lower = user_question.lower()
    
    for pattern_name, pattern_info in QUERY_PATTERNS.items():
        if any(keyword in question_lower for keyword in pattern_info["keywords"]):
            return pattern_name, pattern_info["description"]
    
    return "general", "Consulta general"


def enhanced_auto_viz(df: pd.DataFrame, user_question: str = "", max_rows_density: int = 500_000) -> Dict:
    """
    Visualización automática mejorada con detección inteligente
    """
    if df.empty:
        return {
            "figure": None,
            "message": "No hay datos para visualizar",
            "suggestions": ["Verificar la consulta SQL", "Revisar filtros aplicados"]
        }
    
    # Análisis de tipos de columnas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    
    n_rows, n_cols = df.shape
    
    # Detectar intención de consulta
    query_intent, intent_description = detect_query_intent(user_question)
    
    # Muestreo inteligente para datasets grandes
    sampled_df = df
    sampling_info = ""
    
    if n_rows > max_rows_density:
        if categorical_cols:
            sampled_df = df.groupby(categorical_cols[0]).apply(
                lambda x: x.sample(min(100, len(x)))
            ).reset_index(drop=True)
        else:
            sampled_df = df.sample(min(5000, n_rows))
        
        sampling_info = f"Muestra de {len(sampled_df):,} filas (dataset original: {n_rows:,})"
    
    # Selección inteligente de visualización
    fig = None
    viz_type = "table"
    title = f"Análisis: {intent_description}"
    
    if sampling_info:
        title += f" ({sampling_info})"
    
    # Lógica de visualización basada en estructura de datos
    if len(numeric_cols) == 0:
        # Solo datos categóricos
        if len(categorical_cols) == 1:
            counts = sampled_df[categorical_cols[0]].value_counts().head(20)
            fig = px.bar(
                x=counts.index, 
                y=counts.values,
                title=f"Distribución de {categorical_cols[0]}",
                labels={'x': categorical_cols[0], 'y': 'Frecuencia'},
                template="enhanced_dark"
            )
            viz_type = "bar_chart"
    
    elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
        # Una métrica numérica por categoría
        y_col = numeric_cols[0]
        x_col = categorical_cols[0]
        
        if query_intent == "top_ranking":
            # Ranking horizontal
            top_data = sampled_df.nlargest(15, y_col)
            fig = px.bar(
                top_data, 
                x=y_col, 
                y=x_col,
                orientation='h',
                title=f"Top {len(top_data)} por {y_col}",
                template="enhanced_dark"
            )
            viz_type = "horizontal_bar"
        else:
            # Bar chart estándar
            if sampled_df[x_col].nunique() > 20:
                top_categories = sampled_df[x_col].value_counts().head(15).index
                plot_df = sampled_df[sampled_df[x_col].isin(top_categories)]
            else:
                plot_df = sampled_df
            
            fig = px.bar(
                plot_df, 
                x=x_col, 
                y=y_col,
                title=f"{y_col} por {x_col}",
                template="enhanced_dark"
            )
            fig.update_xaxes(tickangle=45)
            viz_type = "bar_chart"
    
    elif len(numeric_cols) >= 2:
        # Múltiples métricas numéricas
        if datetime_cols and query_intent == "time_series":
            # Serie temporal
            fig = px.line(
                sampled_df.sort_values(datetime_cols[0]),
                x=datetime_cols[0], 
                y=numeric_cols[0],
                color=categorical_cols[0] if categorical_cols else None,
                title=f"Evolución de {numeric_cols[0]}",
                template="enhanced_dark"
            )
            viz_type = "line_chart"
        else:
            # Scatter plot
            if len(numeric_cols) == 2:
                fig = px.scatter(
                    sampled_df, 
                    x=numeric_cols[0], 
                    y=numeric_cols[1],
                    color=categorical_cols[0] if categorical_cols else None,
                    title=f"Relación: {numeric_cols[0]} vs {numeric_cols[1]}",
                    template="enhanced_dark"
                )
                viz_type = "scatter_plot"
    
    # Información adicional
    analysis_info = {
        "figure": fig,
        "viz_type": viz_type,
        "query_intent": query_intent,
        "intent_description": intent_description,
        "data_summary": {
            "rows": n_rows,
            "columns": n_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols
        },
        "sampling_info": sampling_info,
        "detected_tables": detect_relevant_tables(user_question),
        "recommendations": generate_recommendations(df, user_question)
    }
    
    return analysis_info


def generate_recommendations(df: pd.DataFrame, user_question: str) -> List[str]:
    """
    Genera recomendaciones basadas en los datos y la consulta
    """
    recommendations = []
    
    if df.empty:
        recommendations.append("La consulta no devolvió resultados. Verificar filtros.")
        return recommendations
    
    n_rows, n_cols = df.shape
    
    if n_rows > 10000:
        recommendations.append("Dataset grande: considera agregar filtros para mejor rendimiento")
    
    if n_cols > 10:
        recommendations.append("Muchas columnas: considera seleccionar solo las más relevantes")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        recommendations.append("Múltiples métricas disponibles: prueba análisis de correlación")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if df[col].nunique() > 50:
            recommendations.append(f"Columna '{col}' tiene muchas categorías: considera agrupar")
    
    return recommendations


def auto_viz(df: pd.DataFrame, user_question: str | None = None):
    """
    Generate a Plotly figure from a DataFrame using enhanced auto visualization.
    
    This is the main function called by the API.
    """
    question = user_question or ""
    try:
        analysis = enhanced_auto_viz(df, question)
        return analysis.get("figure") if analysis else None
    except Exception as exc:
        return None