import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


pio.templates["my_dark"] = go.layout.Template(
    layout=dict(
        title_font=dict(family="Arial", size=24, color="white"),
        font=dict(family="Arial", size=12, color="white"),
        paper_bgcolor="rgb(17,17,17)",
        plot_bgcolor="rgb(17,17,17)",
        xaxis=dict(gridcolor="#283442", linecolor="#506784", zerolinecolor="#283442"),
        yaxis=dict(gridcolor="#283442", linecolor="#506784", zerolinecolor="#283442")
    )
)

pio.templates.default = "my_dark"

def auto_viz(df: pd.DataFrame, max_rows_density=500_000):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    n_rows, n_cols = df.shape

    if n_rows > max_rows_density:
        if cat_cols:
            df = df.groupby(cat_cols).sample(min(1000, len(df)))
        else:
            df = df.sample(5000)
        title = f"(muestra ~ {n_rows:,} filas)"
    else:
        title = None

    fig = None
    if len(num_cols) == 0:
        fig = None
    elif len(num_cols) == 1 and cat_cols:
        y = num_cols[0]
        x = cat_cols[0]
        fig = px.bar(df, x=x, y=y, template="my_dark", title=title or f"{y} por {x}")
    elif len(num_cols) >= 2:
        fig = px.scatter_matrix(
            df[num_cols + cat_cols[:1]], dimensions=num_cols[:3],
            color=cat_cols[0] if cat_cols else None,
            title=title or "Scatter matrix de variables numéricas",
            template="my_dark"
        )
    elif dt_cols and num_cols:
        fig = px.line(
            df.sort_values(dt_cols[0]),
            x=dt_cols[0], y=num_cols[0],
            color=cat_cols[0] if cat_cols else None,
            title=title or f"Tendencia de {num_cols[0]}",
            template="my_dark"
        )

    return fig

# Ejemplo de uso:
# df = pd.read_sql(sql_string, conn)
# fig = auto_viz(df)
# if fig:
#     fig.show()
# else:
#     display(df.head())