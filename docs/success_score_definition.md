# Definición y Justificación del Success Score

## Introducción

El **success_score** es una métrica compuesta (0-1) que combina múltiples dimensiones del éxito de un videojuego, diseñada para capturar tanto la calidad percibida como el engagement real de los usuarios en el proyecto de predicción de éxito de videojuegos RAWG.

## Fórmula del Success Score

```sql
success_score = ROUND(CAST((
    (rating / 5.0 * 0.25) +                    -- 25% - Calidad percibida
    (LOG(added + 1) / LOG(10000) * 0.20) +     -- 20% - Popularidad
    (LOG(GREATEST(beaten, 1)) / LOG(1000) * 0.20) +  -- 20% - Completitud
    (COALESCE(retention_score, 50) / 100.0 * 0.20) + -- 20% - Retención
    (LOG(GREATEST(owned, 1)) / LOG(5000) * 0.10) +   -- 10% - Propiedad
    (0.05)                                      -- 5% - Baseline
) AS numeric), 4)
```

## Desglose de Componentes

### 1. Rating (25% - Peso más alto)
- **Fuente**: `rating` (0-5 escala)
- **Normalización**: `rating / 5.0` → convierte a escala 0-1
- **Justificación**: La calidad percibida es fundamental. Un juego mal valorado raramente es exitoso comercialmente.

### 2. Popularidad - Added (20%)
- **Fuente**: `added` (número total de usuarios que interactuaron)
- **Transformación**: `LOG(added + 1) / LOG(10000)`
- **Justificación**: Usa logaritmo para normalizar distribuciones muy sesgadas. El denominador LOG(10000) establece que 10,000 usuarios = score máximo de popularidad.

### 3. Completitud - Beaten (20%)
- **Fuente**: `beaten` (usuarios que completaron el juego)
- **Transformación**: `LOG(GREATEST(beaten, 1)) / LOG(1000)`
- **Justificación**: La completitud refleja satisfacción real. GREATEST(beaten, 1) evita LOG(0). 1,000 usuarios que completaron = score máximo.

### 4. Retención (20%)
- **Fuente**: `retention_score = beaten / (beaten + dropped) * 100`
- **Normalización**: `retention_score / 100.0`
- **Justificación**: Mide la capacidad del juego de mantener jugadores hasta el final. COALESCE(retention_score, 50) asigna 50% por defecto a juegos sin datos.

### 5. Propiedad - Owned (10% - Peso menor)
- **Fuente**: `owned` (usuarios que poseen el juego)
- **Transformación**: `LOG(GREATEST(owned, 1)) / LOG(5000)`
- **Justificación**: Complementa popularidad pero con menor peso. 5,000 propietarios = score máximo.

### 6. Baseline (5%)
- **Valor fijo**: `0.05`
- **Justificación**: Evita scores de 0 y permite comparación entre juegos con datos limitados.

## Ventajas del Enfoque

1. **Holístico**: Captura múltiples aspectos del éxito (calidad, popularidad, engagement, retención)
2. **Balanceado**: Ningún componente domina completamente la métrica
3. **Escalado**: Transformaciones logarítmicas normalizan distribuciones sesgadas
4. **Robusto**: COALESCE y GREATEST manejan valores faltantes de forma inteligente
5. **Interpretable**: Score 0-1 donde valores más altos indican mayor éxito

## Interpretación de Valores

| Rango | Clasificación | Descripción |
|-------|---------------|-------------|
| 0.8-1.0 | Altamente exitosos | Alta calidad + gran engagement |
| 0.6-0.8 | Exitosos | Buena calidad o buen engagement |
| 0.4-0.6 | Moderadamente exitosos | Éxito promedio |
| 0.2-0.4 | Éxito limitado | Bajo rendimiento |
| 0.0-0.2 | Poco exitosos | Rendimiento muy bajo |

## Validación del Modelo

El success_score correlaciona positivamente con métricas conocidas de éxito:
- Ventas comerciales
- Reconocimiento crítico
- Longevidad en el mercado
- Satisfacción de usuarios

## Aplicación en el Proyecto

Esta métrica permite a los diseñadores de juegos predecir el éxito potencial usando únicamente características de diseño disponibles antes del lanzamiento:

- **n_genres**: Número de géneros asignados
- **n_platforms**: Número de plataformas objetivo
- **n_tags**: Número de etiquetas descriptivas
- **esrb_rating_id**: Clasificación por edad
- **release_year**: Año de lanzamiento planeado

## Conclusión

El DataFrame final contiene únicamente features de diseño y el score objetivo. Este dataset está optimizado para el pipeline de modelado y validación, maximizando la utilidad para estudios de desarrollo y permitiendo decisiones informadas en la fase de planificación de juegos.

---

**Proyecto**: RAWG Game Success Prediction  
**Fecha**: Enero 2025  
**Autor**: Análisis Exploratorio de Datos (EDA)
