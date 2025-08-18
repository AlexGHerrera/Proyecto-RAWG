"""
ask_visual.py
=============

Módulo optimizado de auto-visualización para datos RAWG.
Genera gráficos Plotly automáticamente basado en estructura de datos.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Optional

# Template oscuro para gráficos
pio.templates["rawg_dark"] = go.layout.Template(
    layout=dict(
        title_font=dict(family="Arial", size=20, color="white"),
        font=dict(family="Arial", size=12, color="#E1E5E9"),
        paper_bgcolor="rgb(17,17,17)",
        plot_bgcolor="rgb(17,17,17)",
        xaxis=dict(gridcolor="#283442", linecolor="#506784", tickcolor="#506784"),
        yaxis=dict(gridcolor="#283442", linecolor="#506784", tickcolor="#506784"),
        colorway=["#00D4FF", "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
    )
)
pio.templates.default = "rawg_dark"





def _is_temporal_query(question: str, columns: list) -> bool:
    """Detecta si es una consulta temporal"""
    question_lower = question.lower()
    has_year_col = any('year' in col.lower() for col in columns)
    has_temporal_words = any(word in question_lower for word in ['year', 'group by year', 'by year', 'temporal'])
    return has_year_col and has_temporal_words


def _create_chart(df: pd.DataFrame, question: str = "") -> Optional[go.Figure]:
    """Crea gráfico automáticamente basado en estructura de datos"""
    if df.empty:
        return None
    
    # Detectar tipos de columnas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Detectar columnas numéricas específicas de RAWG
    rawg_numeric = ['year', 'rating', 'popularity', 'count', 'avg', 'min', 'max', 'sum']
    for col in df.columns:
        if col.lower() in rawg_numeric and col not in numeric_cols:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
            except:
                pass
    
    # Limitar datos para rendimiento
    plot_df = df.head(1000) if len(df) > 1000 else df
    
    # Lógica de visualización simplificada
    try:
        # 1. Series temporales (year + count/rating)
        if _is_temporal_query(question, df.columns) and len(numeric_cols) >= 2:
            year_col = next((col for col in numeric_cols if 'year' in col.lower()), None)
            value_col = next((col for col in numeric_cols if col != year_col), None)
            
            if year_col and value_col:
                return px.line(
                    plot_df.sort_values(year_col),
                    x=year_col, y=value_col,
                    title=f"{value_col} por {year_col}",
                    markers=True,
                    template="rawg_dark"
                )
        
        # 2. Una métrica por categoría
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            
            # Limitar categorías
            if plot_df[x_col].nunique() > 20:
                top_cats = plot_df[x_col].value_counts().head(15).index
                plot_df = plot_df[plot_df[x_col].isin(top_cats)]
            
            return px.bar(
                plot_df, x=x_col, y=y_col,
                title=f"{y_col} por {x_col}",
                template="rawg_dark"
            )
        
        # 3. Solo categóricas - conteo
        if len(categorical_cols) >= 1 and len(numeric_cols) == 0:
            col = categorical_cols[0]
            counts = plot_df[col].value_counts().head(15)
            return px.bar(
                x=counts.index, y=counts.values,
                title=f"Distribución de {col}",
                labels={'x': col, 'y': 'Count'},
                template="rawg_dark"
            )
        
        # 4. Dos variables numéricas
        if len(numeric_cols) >= 2:
            return px.scatter(
                plot_df, x=numeric_cols[0], y=numeric_cols[1],
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                template="rawg_dark"
            )
        
        # 5. Una variable numérica - histograma
        if len(numeric_cols) >= 1:
            return px.histogram(
                plot_df, x=numeric_cols[0],
                title=f"Distribución de {numeric_cols[0]}",
                template="rawg_dark"
            )
            
    except Exception:
        pass
    
    return None




def auto_viz(df: pd.DataFrame, user_question: Optional[str] = None) -> Optional[go.Figure]:
    """
    Genera gráfico Plotly automáticamente basado en estructura de datos.
    Función principal llamada por la API.
    """
    try:
        question = user_question or ""
        fig = _create_chart(df, question)
        return fig
    except Exception:
        return None
