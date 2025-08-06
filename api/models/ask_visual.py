import plotly.express as px
import pandas as pd
import numpy as np


def auto_viz(df: pd.DataFrame, max_rows_density=500_000):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    n_rows, n_cols = df.shape

    # Si hay demasiadas filas, muestrea con estratificación si hay categoría; sino aleatorio
    if n_rows > max_rows_density:
        if cat_cols:
            df = df.groupby(cat_cols).sample(min(1000, len(df)))
        else:
            df = df.sample(5000)
        title = f"(muestra de la visualización — total ≈ {n_rows:,} filas)"
    else:
        title = None

    if len(num_cols) == 0:
        # Solo cualitativas o fechas, renderiza tabla
        fig = None
    elif len(num_cols) == 1 and (len(cat_cols) >= 1 or len(dt_cols) == 0):
        # 1 numérico vs categoría → barras medias
        y = num_cols[0]
        x = cat_cols[0] if cat_cols else None
        if x:
            fig = px.bar(
                df,
                x=x, y=y,
                color=(cat_cols[1] if len(cat_cols) > 1 else None),
                title=title or f"{y} por {x}"
            )
        else:
            fig = px.histogram(df, x=y, title=title or f"Distribución de {y}")
    elif len(num_cols) >= 2:
        # 2 o más numéricos → scatter (o matriz si encontramos variable tema)
        fig = px.scatter_matrix(
            df[num_cols + cat_cols[:1]],
            dimensions=num_cols[:3],
            color=cat_cols[0] if cat_cols else None,
            title=title or "Scatter matrix de variables numéricas"
        )
    elif len(dt_cols) and num_cols:
        # Línea temporal
        fig = px.line(
            df.sort_values(dt_cols[0]),
            x=dt_cols[0], y=num_cols[0],
            color=(cat_cols[0] if cat_cols else None),
            title=title or f"Tendencia de {num_cols[0]} a lo largo del tiempo"
        )
    else:
        fig = None

    return fig