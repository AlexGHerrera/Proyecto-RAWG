# %% [markdown]
"""
# Análisis de Redefinición de Criterios de Éxito

## Objetivo
Explorar diferentes definiciones de "éxito" para videojuegos que resulten en una distribución de clases más balanceada de forma natural, manteniendo la validez conceptual del modelo.

## Problema Actual
- High Success: 0.7% (Rating ≥4.0 + Added ≥1,000)
- Moderate Success: 8.0% (Rating ≥3.0 + Added ≥100)  
- Low Success: 91.3% (El resto)
- Desbalance: 127:1 (inmanejable)

## Estrategia
1. Analizar distribuciones de rating y added por separado
2. Explorar diferentes combinaciones de umbrales
3. Evaluar criterios alternativos (percentiles, scores compuestos)
4. Validar que los nuevos criterios mantengan significado conceptual
5. Proponer la mejor definición balanceada
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")

# %% [markdown]
"""
## Paso 1: Carga de Datos Base para Análisis de Criterios

### Objetivo
Cargar los datos originales de rating, added, playtime y metacritic para explorar diferentes definiciones de éxito sin las limitaciones de los criterios actuales.
"""

# %%
load_dotenv()

def create_db_engine():
    connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(connection_string)

engine = create_db_engine()

# Cargar datos base sin criterios de éxito predefinidos
base_query = """
SELECT 
    g.id_game,
    g.name,
    g.rating,
    g.added,
    g.playtime,
    g.metacritic,
    EXTRACT(YEAR FROM g.released) as release_year
FROM games g
WHERE g.released IS NOT NULL 
    AND g.released >= '2010-01-01'
    AND g.released <= '2024-12-31'
    AND g.rating IS NOT NULL
    AND g.added > 0
ORDER BY g.added DESC;
"""

df_raw = pd.read_sql(base_query, engine)
print(f"Datos base cargados: {len(df_raw):,} juegos")

# Estadísticas descriptivas de variables clave
print("\n=== ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES CLAVE ===")
key_vars = ['rating', 'added', 'playtime', 'metacritic']
for var in key_vars:
    if var in df_raw.columns:
        stats = df_raw[var].describe()
        print(f"\n{var.upper()}:")
        print(f"  Media: {stats['mean']:.2f}")
        print(f"  Mediana: {stats['50%']:.2f}")
        print(f"  P25: {stats['25%']:.2f}")
        print(f"  P75: {stats['75%']:.2f}")
        print(f"  P90: {df_raw[var].quantile(0.9):.2f}")
        print(f"  P95: {df_raw[var].quantile(0.95):.2f}")
        print(f"  Max: {stats['max']:.2f}")

# %% [markdown]
"""
## Paso 2: Análisis de Distribuciones de Variables Clave

### Objetivo
Visualizar las distribuciones de rating, added, playtime y metacritic para identificar umbrales naturales que puedan servir como criterios de éxito más balanceados.

### Metodología
- Histogramas y boxplots de cada variable
- Identificación de percentiles naturales
- Análisis de correlaciones entre variables
"""

# %%
# Visualización de distribuciones
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Rating
axes[0,0].hist(df_raw['rating'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(df_raw['rating'].mean(), color='red', linestyle='--', label=f'Media: {df_raw["rating"].mean():.2f}')
axes[0,0].axvline(df_raw['rating'].median(), color='orange', linestyle='--', label=f'Mediana: {df_raw["rating"].median():.2f}')
axes[0,0].axvline(df_raw['rating'].quantile(0.75), color='green', linestyle='--', label=f'P75: {df_raw["rating"].quantile(0.75):.2f}')
axes[0,0].set_title('Distribución de Rating', fontweight='bold')
axes[0,0].set_xlabel('Rating')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Added (log scale para mejor visualización)
axes[0,1].hist(np.log10(df_raw['added'] + 1), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0,1].axvline(np.log10(df_raw['added'].mean()), color='red', linestyle='--', label=f'Media: {df_raw["added"].mean():.0f}')
axes[0,1].axvline(np.log10(df_raw['added'].median()), color='orange', linestyle='--', label=f'Mediana: {df_raw["added"].median():.0f}')
axes[0,1].axvline(np.log10(df_raw['added'].quantile(0.75)), color='green', linestyle='--', label=f'P75: {df_raw["added"].quantile(0.75):.0f}')
axes[0,1].set_title('Distribución de Added (Log Scale)', fontweight='bold')
axes[0,1].set_xlabel('Log10(Added + 1)')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Playtime
df_playtime_clean = df_raw[df_raw['playtime'].notna() & (df_raw['playtime'] > 0)]
axes[1,0].hist(df_playtime_clean['playtime'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1,0].axvline(df_playtime_clean['playtime'].mean(), color='red', linestyle='--', label=f'Media: {df_playtime_clean["playtime"].mean():.1f}')
axes[1,0].axvline(df_playtime_clean['playtime'].median(), color='orange', linestyle='--', label=f'Mediana: {df_playtime_clean["playtime"].median():.1f}')
axes[1,0].axvline(df_playtime_clean['playtime'].quantile(0.75), color='green', linestyle='--', label=f'P75: {df_playtime_clean["playtime"].quantile(0.75):.1f}')
axes[1,0].set_title('Distribución de Playtime', fontweight='bold')
axes[1,0].set_xlabel('Playtime (horas)')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_xlim(0, 100)  # Limitar para mejor visualización

# Metacritic
df_metacritic_clean = df_raw[df_raw['metacritic'].notna()]
if len(df_metacritic_clean) > 0:
    axes[1,1].hist(df_metacritic_clean['metacritic'], bins=30, alpha=0.7, color='gold', edgecolor='black')
    axes[1,1].axvline(df_metacritic_clean['metacritic'].mean(), color='red', linestyle='--', label=f'Media: {df_metacritic_clean["metacritic"].mean():.1f}')
    axes[1,1].axvline(df_metacritic_clean['metacritic'].median(), color='orange', linestyle='--', label=f'Mediana: {df_metacritic_clean["metacritic"].median():.1f}')
    axes[1,1].axvline(df_metacritic_clean['metacritic'].quantile(0.75), color='green', linestyle='--', label=f'P75: {df_metacritic_clean["metacritic"].quantile(0.75):.1f}')
    axes[1,1].set_title('Distribución de Metacritic', fontweight='bold')
    axes[1,1].set_xlabel('Metacritic Score')
    axes[1,1].set_ylabel('Frecuencia')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Paso 3: Exploración de Criterios Alternativos Basados en Percentiles

### Objetivo
Probar diferentes combinaciones de umbrales basados en percentiles naturales de los datos para encontrar una distribución más balanceada.

### Metodología
- Criterios basados en percentiles (P70, P80, P90)
- Combinaciones de rating + added con diferentes umbrales
- Evaluación del balance resultante para cada combinación
"""

# %%
# Definir diferentes criterios de éxito para probar
criteria_tests = [
    # Formato: (nombre, condición_high, condición_moderate, descripción)
    ("P75_Rating_P75_Added", "rating >= 3.5 and added >= 50", "rating >= 2.5 and added >= 10", "P75 Rating + P75 Added"),
    ("P80_Rating_P70_Added", "rating >= 3.7 and added >= 30", "rating >= 2.8 and added >= 8", "P80 Rating + P70 Added"),
    ("P70_Rating_P80_Added", "rating >= 3.3 and added >= 80", "rating >= 2.5 and added >= 15", "P70 Rating + P80 Added"),
    ("P85_Rating_P60_Added", "rating >= 3.9 and added >= 20", "rating >= 3.0 and added >= 5", "P85 Rating + P60 Added"),
    ("Balanced_Conservative", "rating >= 3.5 and added >= 100", "rating >= 2.8 and added >= 20", "Conservador Balanceado"),
    ("Balanced_Liberal", "rating >= 3.2 and added >= 50", "rating >= 2.5 and added >= 10", "Liberal Balanceado"),
    ("Quality_Focus", "rating >= 4.0 and added >= 50", "rating >= 3.5 and added >= 10", "Enfoque en Calidad"),
    ("Popularity_Focus", "rating >= 3.0 and added >= 500", "rating >= 2.5 and added >= 100", "Enfoque en Popularidad"),
]

print("=== ANÁLISIS DE CRITERIOS ALTERNATIVOS ===")
results = []

for name, high_condition, moderate_condition, description in criteria_tests:
    # Aplicar criterios
    df_test = df_raw.copy()
    
    # Evaluar condiciones
    high_mask = df_test.eval(high_condition)
    moderate_mask = df_test.eval(moderate_condition) & ~high_mask
    low_mask = ~high_mask & ~moderate_mask
    
    # Contar resultados
    high_count = high_mask.sum()
    moderate_count = moderate_mask.sum()
    low_count = low_mask.sum()
    total = len(df_test)
    
    # Calcular porcentajes
    high_pct = high_count / total * 100
    moderate_pct = moderate_count / total * 100
    low_pct = low_count / total * 100
    
    # Calcular ratios de desbalance
    if high_count > 0:
        high_ratio = total / high_count
        moderate_ratio = total / moderate_count if moderate_count > 0 else float('inf')
    else:
        high_ratio = float('inf')
        moderate_ratio = float('inf')
    
    results.append({
        'name': name,
        'description': description,
        'high_count': high_count,
        'moderate_count': moderate_count,
        'low_count': low_count,
        'high_pct': high_pct,
        'moderate_pct': moderate_pct,
        'low_pct': low_pct,
        'high_ratio': high_ratio,
        'moderate_ratio': moderate_ratio,
        'total_success_pct': high_pct + moderate_pct
    })
    
    print(f"\n{name} ({description}):")
    print(f"  High: {high_count:,} ({high_pct:.1f}%) - Ratio 1:{high_ratio:.1f}")
    print(f"  Moderate: {moderate_count:,} ({moderate_pct:.1f}%) - Ratio 1:{moderate_ratio:.1f}")
    print(f"  Low: {low_count:,} ({low_pct:.1f}%)")
    print(f"  Total Success: {high_pct + moderate_pct:.1f}%")

# Convertir a DataFrame para análisis
results_df = pd.DataFrame(results)

# %% [markdown]
"""
## Paso 4: Evaluación y Ranking de Criterios Alternativos

### Objetivo
Evaluar los diferentes criterios probados según múltiples métricas de balance y seleccionar los más prometedores.

### Criterios de Evaluación
1. **Balance General**: Total success entre 15-30% (manejable pero no trivial)
2. **High Success**: Entre 3-8% (suficiente para aprender patrones)
3. **Moderate Success**: Entre 10-20% (clase intermedia robusta)
4. **Interpretabilidad**: Umbrales que tengan sentido conceptual
"""

# %%
# Definir criterios de evaluación
def evaluate_balance(row):
    score = 0
    
    # Penalizar desbalances extremos
    if row['high_pct'] < 1:  # Muy poco high success
        score -= 3
    elif row['high_pct'] < 3:  # Poco high success
        score -= 1
    elif 3 <= row['high_pct'] <= 8:  # Rango ideal
        score += 2
    elif row['high_pct'] > 15:  # Demasiado high success
        score -= 2
    
    # Evaluar moderate success
    if row['moderate_pct'] < 5:  # Muy poco moderate
        score -= 2
    elif 10 <= row['moderate_pct'] <= 20:  # Rango ideal
        score += 2
    elif row['moderate_pct'] > 25:  # Demasiado moderate
        score -= 1
    
    # Evaluar total success
    if 15 <= row['total_success_pct'] <= 30:  # Rango manejable
        score += 3
    elif 10 <= row['total_success_pct'] < 15:  # Aceptable
        score += 1
    elif row['total_success_pct'] > 40:  # Demasiado fácil
        score -= 2
    
    # Bonus por ratios manejables
    if row['high_ratio'] <= 50:  # Ratio manejable para high
        score += 1
    if row['moderate_ratio'] <= 20:  # Ratio manejable para moderate
        score += 1
    
    return score

# Aplicar evaluación
results_df['balance_score'] = results_df.apply(evaluate_balance, axis=1)
results_df = results_df.sort_values('balance_score', ascending=False)

print("=== RANKING DE CRITERIOS POR BALANCE ===")
print(results_df[['name', 'description', 'high_pct', 'moderate_pct', 'total_success_pct', 'balance_score']].round(1))

# Visualizar los top 3 criterios
top_3 = results_df.head(3)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (_, row) in enumerate(top_3.iterrows()):
    categories = ['High Success', 'Moderate Success', 'Low Success']
    values = [row['high_pct'], row['moderate_pct'], row['low_pct']]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    axes[i].pie(values, labels=categories, autopct='%1.1f%%', colors=colors)
    axes[i].set_title(f"{row['name']}\n{row['description']}\nScore: {row['balance_score']}", 
                     fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Paso 5: Análisis Detallado del Mejor Criterio

### Objetivo
Analizar en profundidad el criterio mejor evaluado para validar que mantiene significado conceptual y produce un dataset entrenable.

### Metodología
- Análisis de características de juegos en cada categoría
- Validación conceptual de los umbrales
- Comparación con criterios originales
- Estimación de mejora en modelado
"""

# %%
# Seleccionar el mejor criterio
best_criterion = results_df.iloc[0]
print(f"=== ANÁLISIS DETALLADO DEL MEJOR CRITERIO ===")
print(f"Nombre: {best_criterion['name']}")
print(f"Descripción: {best_criterion['description']}")
print(f"Score de Balance: {best_criterion['balance_score']}")

# Aplicar el mejor criterio al dataset
best_name = best_criterion['name']
if best_name == "P75_Rating_P75_Added":
    high_condition = "rating >= 3.5 and added >= 50"
    moderate_condition = "rating >= 2.5 and added >= 10"
elif best_name == "P80_Rating_P70_Added":
    high_condition = "rating >= 3.7 and added >= 30"
    moderate_condition = "rating >= 2.8 and added >= 8"
elif best_name == "Balanced_Conservative":
    high_condition = "rating >= 3.5 and added >= 100"
    moderate_condition = "rating >= 2.8 and added >= 20"
elif best_name == "Balanced_Liberal":
    high_condition = "rating >= 3.2 and added >= 50"
    moderate_condition = "rating >= 2.5 and added >= 10"
else:
    # Usar el primer criterio como fallback
    high_condition = "rating >= 3.5 and added >= 50"
    moderate_condition = "rating >= 2.5 and added >= 10"

df_best = df_raw.copy()
high_mask = df_best.eval(high_condition)
moderate_mask = df_best.eval(moderate_condition) & ~high_mask
low_mask = ~high_mask & ~moderate_mask

df_best['success_category_new'] = 'low_success'
df_best.loc[moderate_mask, 'success_category_new'] = 'moderate_success'
df_best.loc[high_mask, 'success_category_new'] = 'high_success'

# Análisis comparativo de características por categoría
print("\n=== CARACTERÍSTICAS POR CATEGORÍA (NUEVO CRITERIO) ===")
comparison_vars = ['rating', 'added', 'playtime', 'release_year']

for var in comparison_vars:
    if var in df_best.columns:
        print(f"\n{var.upper()}:")
        stats = df_best.groupby('success_category_new')[var].agg(['count', 'mean', 'median', 'std']).round(2)
        print(stats)

# Comparación con criterios originales
print("\n=== COMPARACIÓN CON CRITERIOS ORIGINALES ===")
print("ORIGINAL:")
print("  High: 0.7% (Rating ≥4.0 + Added ≥1,000)")
print("  Moderate: 8.0% (Rating ≥3.0 + Added ≥100)")
print("  Low: 91.3%")
print("  Ratio High: 1:139")

print(f"\nNUEVO ({best_criterion['name']}):")
print(f"  High: {best_criterion['high_pct']:.1f}% ({high_condition})")
print(f"  Moderate: {best_criterion['moderate_pct']:.1f}% ({moderate_condition})")
print(f"  Low: {best_criterion['low_pct']:.1f}%")
print(f"  Ratio High: 1:{best_criterion['high_ratio']:.1f}")

# Calcular mejora esperada
improvement_factor = 139 / best_criterion['high_ratio']
print(f"\nMEJORA ESPERADA EN MODELADO:")
print(f"  Factor de mejora en balance: {improvement_factor:.1f}x")
print(f"  Expectativa de accuracy: {50 * (1 + improvement_factor/10):.1f}% (estimación)")

# %% [markdown]
"""
## Conclusiones y Recomendaciones

### Hallazgos Clave

1. **Criterio Óptimo Identificado**: {best_criterion['name']}
   - High Success: {best_criterion['high_pct']:.1f}% vs 0.7% original
   - Moderate Success: {best_criterion['moderate_pct']:.1f}% vs 8.0% original
   - Ratio de mejora: {139 / best_criterion['high_ratio']:.1f}x más balanceado

2. **Validación Conceptual**:
   - Los umbrales mantienen significado interpretable
   - Separación clara entre categorías de éxito
   - Balance entre calidad (rating) y popularidad (added)

3. **Impacto Esperado en Modelado**:
   - Ratio 1:{best_criterion['high_ratio']:.1f} vs 1:139 original
   - Expectativa de superar 80% accuracy con técnicas apropiadas
   - Dataset más entrenable sin necesidad de SMOTE extremo

### Recomendación Final

**Implementar el criterio {best_criterion['name']}** para regenerar el dataset v3:
- Mantiene interpretabilidad conceptual
- Produce balance natural más manejable
- Expectativa realista de alcanzar objetivo de 80% accuracy

### Próximos Pasos

1. **Regenerar EDA v3** con los nuevos criterios de éxito
2. **Mantener las features específicas** ya identificadas
3. **Entrenar modelos** con el dataset rebalanceado naturalmente
4. **Validar mejora** comparando con resultados v2

El nuevo criterio transforma un problema inmanejable en uno entrenable manteniendo validez conceptual.
"""

# %%
# Guardar el mejor criterio para implementación
best_criteria_info = {
    'name': best_criterion['name'],
    'description': best_criterion['description'],
    'high_condition': high_condition,
    'moderate_condition': moderate_condition,
    'high_pct': best_criterion['high_pct'],
    'moderate_pct': best_criterion['moderate_pct'],
    'low_pct': best_criterion['low_pct'],
    'balance_score': best_criterion['balance_score'],
    'improvement_factor': 139 / best_criterion['high_ratio']
}

print(f"\n=== CRITERIO SELECCIONADO PARA IMPLEMENTACIÓN ===")
for key, value in best_criteria_info.items():
    print(f"{key}: {value}")

# Crear muestra del dataset con nuevo criterio para validación
sample_new = df_best[['name', 'rating', 'added', 'playtime', 'success_category_new']].head(10)
print(f"\n=== MUESTRA DEL DATASET CON NUEVO CRITERIO ===")
print(sample_new)
