# %% [markdown]
"""
# Análisis de Correlaciones y Distribuciones - Dataset v3

## Objetivo
Analizar en profundidad el dataset generado por el EDA v3 para:
1. Identificar el problema del desbalance extremo de clases
2. Analizar correlaciones entre features y target
3. Evaluar distribuciones de variables
4. Proponer soluciones concretas para el desbalance
5. Redefinir criterios de éxito si es necesario

## Problema Identificado
- High Success: 0.7% (461 juegos)
- Moderate Success: 8.0% (5,103 juegos) 
- Low Success: 91.3% (58,551 juegos)

Este desbalance extremo explica el ceiling de ~50% accuracy en versiones anteriores.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")

# %% [markdown]
"""
## Paso 1: Carga y Exploración Inicial del Dataset

### Objetivo
Cargar el dataset v3 y realizar una exploración inicial para confirmar el problema de desbalance y entender la estructura de los datos.
"""

# %%
# Cargar dataset v3
df = pd.read_csv('../data/classification_dataset_v3.csv')

print("=== INFORMACIÓN BÁSICA DEL DATASET ===")
print(f"Dimensiones: {df.shape}")
print(f"Columnas: {len(df.columns)}")
print(f"Filas: {len(df):,}")

print("\n=== DISTRIBUCIÓN DE LA VARIABLE OBJETIVO ===")
target_dist = df['success_category'].value_counts()
target_pct = df['success_category'].value_counts(normalize=True) * 100

for category in target_dist.index:
    count = target_dist[category]
    pct = target_pct[category]
    print(f"{category}: {count:,} ({pct:.1f}%)")

print("\n=== INFORMACIÓN DE COLUMNAS ===")
print(df.info())

# %% [markdown]
"""
## Paso 2: Análisis Detallado del Desbalance de Clases

### Objetivo
Visualizar y cuantificar el problema del desbalance para entender su magnitud y proponer soluciones específicas.

### Metodología
- Visualización de distribución de clases
- Cálculo de ratios de desbalance
- Análisis de implicaciones para modelado
"""

# %%
# Visualización del desbalance
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de barras
target_dist.plot(kind='bar', ax=axes[0], color=['#ff7f7f', '#ffb347', '#90ee90'])
axes[0].set_title('Distribución de Categorías de Éxito', fontweight='bold')
axes[0].set_xlabel('Categoría de Éxito')
axes[0].set_ylabel('Número de Juegos')
axes[0].tick_params(axis='x', rotation=45)

# Añadir etiquetas con porcentajes
for i, (category, count) in enumerate(target_dist.items()):
    pct = target_pct[category]
    axes[0].text(i, count + 1000, f'{count:,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

# Gráfico de pastel
axes[1].pie(target_dist.values, labels=target_dist.index, autopct='%1.1f%%',
           colors=['#ff7f7f', '#ffb347', '#90ee90'])
axes[1].set_title('Proporción de Categorías de Éxito', fontweight='bold')

plt.tight_layout()
plt.show()

# Cálculos de desbalance
print("=== ANÁLISIS DE DESBALANCE ===")
total_games = len(df)
high_success_ratio = target_dist['high_success'] / total_games
moderate_success_ratio = target_dist['moderate_success'] / total_games
low_success_ratio = target_dist['low_success'] / total_games

print(f"Ratio High Success: 1:{int(1/high_success_ratio)}")
print(f"Ratio Moderate Success: 1:{int(1/moderate_success_ratio)}")
print(f"Ratio Low Success: 1:{int(1/low_success_ratio)}")

print(f"\nClase minoritaria (high_success): {high_success_ratio:.3f} ({high_success_ratio*100:.1f}%)")
print(f"Desbalance extremo: {low_success_ratio/high_success_ratio:.1f}:1")

# %% [markdown]
"""
## Paso 3: Análisis de Correlaciones entre Features

### Objetivo
Identificar qué features tienen mayor correlación con el éxito y detectar posibles redundancias entre variables.

### Metodología
- Matriz de correlación completa
- Correlaciones específicas con la variable objetivo
- Identificación de features más predictivas
"""

# %%
# Preparar datos para correlación (convertir categórica a numérica)
df_corr = df.copy()
df_corr['success_numeric'] = df_corr['success_category'].map({
    'low_success': 0,
    'moderate_success': 1, 
    'high_success': 2
})

# Seleccionar solo columnas numéricas para correlación
numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('success_numeric')  # Remover para análisis separado

# Matriz de correlación
corr_matrix = df_corr[numeric_cols + ['success_numeric']].corr()

# Visualización de matriz de correlación
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación - Features vs Success', fontweight='bold')
plt.tight_layout()
plt.show()

# Correlaciones específicas con success
success_correlations = corr_matrix['success_numeric'].drop('success_numeric').sort_values(key=abs, ascending=False)

print("=== TOP 10 CORRELACIONES CON ÉXITO ===")
for feature, corr in success_correlations.head(10).items():
    print(f"{feature}: {corr:.3f}")

print("\n=== CORRELACIONES NEGATIVAS MÁS FUERTES ===")
negative_corrs = success_correlations[success_correlations < 0].head(5)
for feature, corr in negative_corrs.items():
    print(f"{feature}: {corr:.3f}")

# %% [markdown]
"""
## Paso 4: Análisis de Distribuciones de Features Clave

### Objetivo
Analizar las distribuciones de las features más correlacionadas con el éxito para entender patrones y posibles transformaciones.

### Metodología
- Histogramas de features numéricas clave
- Análisis de features binarias por categoría de éxito
- Identificación de patrones discriminativos
"""

# %%
# Análisis de features numéricas clave
numeric_features = ['n_genres', 'n_platforms', 'n_tags', 'release_year', 'playtime']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(numeric_features):
    if i < len(axes):
        # Histograma por categoría de éxito
        for category in df['success_category'].unique():
            subset = df[df['success_category'] == category][feature].dropna()
            axes[i].hist(subset, alpha=0.6, label=category, bins=20)
        
        axes[i].set_title(f'Distribución de {feature}', fontweight='bold')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frecuencia')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

# Remover subplot vacío
if len(numeric_features) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()

# Estadísticas descriptivas por categoría
print("=== ESTADÍSTICAS DESCRIPTIVAS POR CATEGORÍA DE ÉXITO ===")
for feature in numeric_features:
    print(f"\n--- {feature.upper()} ---")
    stats = df.groupby('success_category')[feature].agg(['count', 'mean', 'median', 'std']).round(2)
    print(stats)

# %% [markdown]
"""
## Paso 5: Análisis de Features Binarias Específicas

### Objetivo
Analizar el comportamiento de las features binarias (géneros, plataformas, tags específicos) para identificar cuáles son más discriminativas.

### Metodología
- Análisis de frecuencia de features binarias por categoría
- Test de chi-cuadrado para significancia estadística
- Identificación de features más predictivas
"""

# %%
# Identificar features binarias
binary_features = [col for col in df.columns if col.startswith('is_')]

# Análisis de features binarias más discriminativas
print("=== ANÁLISIS DE FEATURES BINARIAS ===")
binary_analysis = []

for feature in binary_features:
    # Tabla de contingencia
    contingency = pd.crosstab(df[feature], df['success_category'])
    
    # Test chi-cuadrado
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Porcentajes por categoría
    percentages = pd.crosstab(df[feature], df['success_category'], normalize='columns') * 100
    
    # Guardar resultados
    binary_analysis.append({
        'feature': feature,
        'chi2': chi2,
        'p_value': p_value,
        'high_success_pct': percentages.loc[1, 'high_success'] if 1 in percentages.index else 0,
        'moderate_success_pct': percentages.loc[1, 'moderate_success'] if 1 in percentages.index else 0,
        'low_success_pct': percentages.loc[1, 'low_success'] if 1 in percentages.index else 0
    })

# Convertir a DataFrame y ordenar por chi-cuadrado
binary_df = pd.DataFrame(binary_analysis)
binary_df = binary_df.sort_values('chi2', ascending=False)

print("TOP 10 FEATURES BINARIAS MÁS DISCRIMINATIVAS:")
print(binary_df.head(10)[['feature', 'chi2', 'p_value', 'high_success_pct', 'moderate_success_pct', 'low_success_pct']].round(3))

# Visualización de top features binarias
top_binary_features = binary_df.head(6)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(top_binary_features):
    contingency = pd.crosstab(df[feature], df['success_category'])
    percentages = pd.crosstab(df[feature], df['success_category'], normalize='columns') * 100
    
    percentages.T.plot(kind='bar', ax=axes[i], color=['lightcoral', 'lightblue'])
    axes[i].set_title(f'{feature}\n(Chi2: {binary_df[binary_df.feature==feature].chi2.iloc[0]:.1f})', 
                     fontweight='bold')
    axes[i].set_xlabel('Categoría de Éxito')
    axes[i].set_ylabel('Porcentaje')
    axes[i].legend(['No tiene', 'Tiene'], loc='upper right')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Paso 6: Propuestas Concretas para Solucionar el Desbalance

### Objetivo
Basándose en el análisis realizado, proponer soluciones específicas para el problema de desbalance que permitan alcanzar el objetivo de 80% accuracy.

### Metodología
- Evaluar diferentes estrategias de rebalanceo
- Proponer nuevos criterios de definición de éxito
- Recomendar técnicas de modelado apropiadas
"""

# %%
print("=== PROPUESTAS PARA SOLUCIONAR EL DESBALANCE ===")

print("\n1. REFORMULACIÓN DEL TARGET (RECOMENDADO)")
print("   Opción A: Target Binario")
print("   - Success (moderate + high): 8.7%")
print("   - No Success (low): 91.3%")
print("   - Ratio: 1:10.5 (más manejable que 1:130)")

print("\n   Opción B: Nuevos Criterios de Éxito")
print("   - Bajar umbrales: Rating ≥3.5 + Added ≥500")
print("   - Crear distribución más balanceada naturalmente")

# Simular target binario
df['success_binary'] = (df['success_category'].isin(['moderate_success', 'high_success'])).astype(int)
binary_dist = df['success_binary'].value_counts()
binary_pct = df['success_binary'].value_counts(normalize=True) * 100

print(f"\n   SIMULACIÓN TARGET BINARIO:")
print(f"   No Success (0): {binary_dist[0]:,} ({binary_pct[0]:.1f}%)")
print(f"   Success (1): {binary_dist[1]:,} ({binary_pct[1]:.1f}%)")
print(f"   Ratio: 1:{binary_dist[0]/binary_dist[1]:.1f}")

print("\n2. TÉCNICAS DE BALANCEO DE CLASES")
print("   - SMOTE: Generar ejemplos sintéticos de clase minoritaria")
print("   - Class Weights: Penalizar más errores en clase minoritaria")
print("   - Undersampling: Reducir clase mayoritaria (no recomendado)")
print("   - Ensemble Methods: Combinar múltiples modelos balanceados")

print("\n3. MÉTRICAS DE EVALUACIÓN APROPIADAS")
print("   - ROC-AUC: Mejor para datasets desbalanceados")
print("   - F1-Score: Balance entre precision y recall")
print("   - Precision-Recall AUC: Foco en clase minoritaria")
print("   - Balanced Accuracy: Promedio de sensitividad por clase")

print("\n4. ENRIQUECIMIENTO DE FEATURES")
print("   - Features derivadas: ratios, interacciones")
print("   - Features temporales: tendencias por año")
print("   - Features de complejidad: combinaciones de categorías")

# Análisis de correlación con target binario
binary_corr = df[numeric_cols + ['success_binary']].corr()['success_binary'].drop('success_binary')
binary_corr = binary_corr.sort_values(key=abs, ascending=False)

print("\n=== CORRELACIONES CON TARGET BINARIO ===")
print("Top 5 correlaciones positivas:")
for feature, corr in binary_corr.head(5).items():
    print(f"  {feature}: {corr:.3f}")

print("\nTop 5 correlaciones negativas:")
negative_binary = binary_corr[binary_corr < 0].head(5)
for feature, corr in negative_binary.items():
    print(f"  {feature}: {corr:.3f}")

# %% [markdown]
"""
## Conclusiones y Recomendaciones

### Hallazgos Clave

1. **Desbalance Extremo Confirmado**: 
   - Ratio 1:130 para high_success es inmanejable
   - Explica directamente el ceiling de ~50% accuracy

2. **Features Más Predictivas Identificadas**:
   - Variables numéricas: playtime, n_tags, release_year
   - Features binarias: Géneros y plataformas específicas con alta significancia estadística

3. **Patrones Discriminativos**:
   - Juegos exitosos tienden a tener más tags y mayor playtime
   - Ciertas combinaciones de géneros/plataformas son más predictivas

### Recomendación Principal

**Reformular el target a clasificación binaria**:
- Success: moderate_success + high_success (8.7%)
- No Success: low_success (91.3%)
- Ratio 1:10.5 es mucho más manejable

### Próximos Pasos Sugeridos

1. **Implementar target binario** y regenerar dataset
2. **Aplicar SMOTE** para balancear clases
3. **Entrenar modelos** con class weights apropiados
4. **Usar métricas correctas**: ROC-AUC, F1-Score
5. **Validar** con estratificación de clases

Con estas modificaciones, el objetivo de 80% accuracy se vuelve alcanzable.
"""
