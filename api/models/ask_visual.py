import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def auto_viz(df: pd.DataFrame, max_rows_density: int = 500_000) -> Optional[go.Figure]:
    """Genera visualización automática inteligente basada en tipos de datos"""
    try:
        # Validar DataFrame
        if df.empty:
            logger.warning("DataFrame vacío recibido")
            return None
        
        # Detectar tipos de columnas
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        n_rows, n_cols_total = df.shape
        
        logger.info(f"Datos: {n_rows} filas, {len(num_cols)} numéricas, {len(cat_cols)} categóricas, {len(dt_cols)} fechas")

        # Muestreo inteligente para datasets grandes
        sample_title = ""
        if n_rows > max_rows_density:
            if cat_cols:
                # Muestreo estratificado por categorías
                try:
                    df = df.groupby(cat_cols[0]).apply(lambda x: x.sample(min(1000, len(x)))).reset_index(drop=True)
                except:
                    df = df.sample(min(5000, n_rows))
            else:
                df = df.sample(min(5000, n_rows))
            sample_title = f" (muestra de {len(df):,} de {n_rows:,} filas)"

        # Lógica de visualización mejorada
        fig = None
        
        # 1. Solo datos categóricos → gráfico de barras de conteos
        if len(num_cols) == 0 and len(cat_cols) > 0:
            main_cat = cat_cols[0]
            counts = df[main_cat].value_counts().head(20)  # Top 20 para evitar sobrecarga
            fig = px.bar(
                x=counts.index, y=counts.values,
                title=f"Distribución de {main_cat}{sample_title}",
                labels={'x': main_cat, 'y': 'Cantidad'}
            )
            
        # 2. Una variable numérica + categorías → barras agrupadas o boxplot
        elif len(num_cols) == 1 and len(cat_cols) >= 1:
            y_col = num_cols[0]
            x_col = cat_cols[0]
            
            # Si hay muchas categorías, usar boxplot; sino barras
            unique_cats = df[x_col].nunique()
            if unique_cats > 15:
                fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} por {x_col}{sample_title}")
            else:
                # Agregar por media para barras
                agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                color_col = cat_cols[1] if len(cat_cols) > 1 else None
                fig = px.bar(
                    agg_df, x=x_col, y=y_col,
                    title=f"Promedio de {y_col} por {x_col}{sample_title}"
                )
                
        # 3. Solo una variable numérica → histograma
        elif len(num_cols) == 1 and len(cat_cols) == 0:
            y_col = num_cols[0]
            fig = px.histogram(df, x=y_col, title=f"Distribución de {y_col}{sample_title}")
            
        # 4. Dos variables numéricas → scatter plot
        elif len(num_cols) == 2:
            x_col, y_col = num_cols[0], num_cols[1]
            color_col = cat_cols[0] if cat_cols else None
            size_col = num_cols[2] if len(num_cols) > 2 else None
            
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col, size=size_col,
                title=f"{y_col} vs {x_col}{sample_title}"
            )
            
        # 5. Múltiples variables numéricas → matriz de correlación o scatter matrix
        elif len(num_cols) >= 3:
            if len(num_cols) <= 5:  # Scatter matrix para pocas variables
                fig = px.scatter_matrix(
                    df, dimensions=num_cols[:4],
                    color=cat_cols[0] if cat_cols else None,
                    title=f"Matriz de correlaciones{sample_title}"
                )
            else:  # Heatmap de correlación para muchas variables
                corr_matrix = df[num_cols].corr()
                fig = px.imshow(
                    corr_matrix, text_auto=True, aspect="auto",
                    title=f"Mapa de correlaciones{sample_title}"
                )
                
        # 6. Series temporales
        elif len(dt_cols) >= 1 and len(num_cols) >= 1:
            time_col = dt_cols[0]
            y_col = num_cols[0]
            color_col = cat_cols[0] if cat_cols else None
            
            df_sorted = df.sort_values(time_col)
            fig = px.line(
                df_sorted, x=time_col, y=y_col, color=color_col,
                title=f"Evolución de {y_col} en el tiempo{sample_title}"
            )
            
        # 7. Caso especial: datos de videojuegos (rating, metacritic, etc.)
        if fig is None and any(col in df.columns for col in ['rating', 'metacritic', 'name']):
            if 'rating' in df.columns and 'name' in df.columns:
                top_games = df.nlargest(20, 'rating')
                fig = px.bar(
                    top_games, x='name', y='rating',
                    title=f"Top 20 juegos por rating{sample_title}"
                )
                fig.update_xaxes(tickangle=45)
        
        # Mejoras estéticas generales
        if fig:
            fig.update_layout(
                template="plotly_white",
                font=dict(size=12),
                title_font_size=16,
                showlegend=True if len(cat_cols) > 0 else False
            )
            logger.info("Visualización generada exitosamente")
        else:
            logger.warning("No se pudo determinar tipo de visualización apropiado")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error generando visualización: {str(e)}")
        return None