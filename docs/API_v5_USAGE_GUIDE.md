# RAWG API v5 - Guía de Uso y Documentación

[![API Version](https://img.shields.io/badge/API-v5.0.0-blue)](http://51.20.113.231/docs)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](http://51.20.113.231/health)
[![Model](https://img.shields.io/badge/Model-T5%20Ultra--Optimized-orange)](http://51.20.113.231/model/info)
[![Success Rate](https://img.shields.io/badge/Success%20Rate-90%25+-brightgreen)](#-patrones-exitosos-90-éxito)

## Descripción General

La **RAWG API v5** es un sistema ultra-optimizado de consultas en lenguaje natural que convierte preguntas sobre videojuegos en consultas SQL precisas, genera visualizaciones automáticas y predice el éxito de videojuegos usando características de diseño.

### Características Principales

- **NL→SQL Ultra-Optimizado**: Modelo T5 fine-tuned con schema específicamente diseñado para máxima compatibilidad
- **Vista Optimizada**: Esquema `games_optimized` con 860k+ juegos y columnas optimizadas para T5
- **Corrección Inteligente**: Sistema de post-procesamiento basado en patrones de fallos identificados
- **Visualizaciones Automáticas**: Gráficos Plotly generados dinámicamente con detección inteligente
- **Predicción ML**: Modelo de regresión para éxito de videojuegos
- **Patrones Validados**: 90%+ de éxito con sintaxis WHERE optimizada
- **Seguridad**: Sanitización SQL y validación de entrada

---

## Endpoints Principales

### 1. `/ask-text` - Consultas de Texto
**POST** - Convierte lenguaje natural a SQL y devuelve resultados

```bash
curl -X POST "http://localhost:8000/ask-text" \
  -H "Content-Type: application/json" \
  -d '{"question": "top 10 juegos por rating"}'
```

**Respuesta:**
```json
{
  "question": "top 10 juegos por rating",
  "sql": "SELECT name, rating FROM games_complete ORDER BY rating DESC LIMIT 10",
  "data": [
    {"name": "The Witcher 3", "rating": 4.66},
    {"name": "Portal 2", "rating": 4.62}
  ],
  "metadata": {"execution_time_ms": 1234, "rows_returned": 10},
  "execution_time": 1.23,
  "success": true
}
```

### 2. `/ask-visual` - Consultas con Visualización JSON
**POST** - Genera consulta SQL + visualización Plotly como JSON

```bash
curl -X POST "http://localhost:8000/ask-visual" \
  -H "Content-Type: application/json" \
  -d '{"question": "distribución de juegos por plataforma"}'
```

**Respuesta:**
```json
{
  "question": "distribución de juegos por plataforma",
  "sql": "SELECT platform, COUNT(*) as count FROM games_complete GROUP BY platform",
  "data": [
    {"platform": "PC", "count": 45000},
    {"platform": "PlayStation", "count": 32000}
  ],
  "visualization": {
    "data": [{"type": "bar", "x": ["PC", "PlayStation"], "y": [45000, 32000]}],
    "layout": {"title": "Distribución de juegos por plataforma"}
  },
  "execution_time": 2.1,
  "success": true
}
```

### 3. `/ask-visual-html` - Visualización HTML
**POST** - Devuelve visualización como HTML listo para navegador

```bash
curl -X POST "http://localhost:8000/ask-visual-html" \
  -H "Content-Type: application/json" \
  -d '{"question": "mejores géneros por rating promedio"}'
```

**Respuesta:** HTML completo con gráfico Plotly embebido

### 4. `/predict` - Predicción de Éxito
**POST** - Predice éxito de videojuego usando características de diseño

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "genres": ["rpg", "action"],
    "platforms": ["pc", "playstation 4"],
    "tags": ["multiplayer", "adventure"],
    "estimated_hours": 30.0,
    "release_year": 2024
  }'
```

**Respuesta:**
```json
{
  "predicted_class": "high_success",
  "confidence": 0.85,
  "probabilities": {
    "high_success": 0.85,
    "moderate_success": 0.12,
    "low_success": 0.03
  },
  "execution_time": 0.45,
  "success": true
}
```

---

## Endpoints de Utilidad

### `/health` - Health Check
```bash
curl http://localhost:8000/health
```

### `/model/info` - Información del Modelo
```bash
curl http://localhost:8000/model/info
```

### `/test-model` - Test de Conectividad
```bash
curl http://localhost:8000/test-model
```

### `/docs` - Documentación Interactiva
Abrir en navegador: `http://localhost:8000/docs`

---

## 🎯 Patrones Exitosos (90%+ Éxito)

> **IMPORTANTE**: Estos patrones han sido validados con tests exhaustivos y garantizan alta tasa de éxito.

### ✅ Consultas Básicas Optimizadas

```sql
-- Filtros por año (100% éxito)
"games where year equals 2020"
"games where year equals 2019"
"games where year equals 2021"

-- Filtros por rating (100% éxito)
"games where rating above 4"
"games where rating above 3"
"games where rating equals 5"
"games where rating greater than 4.5"

-- Filtros por popularidad (100% éxito)
"games where popularity above 100"
"games where popularity above 50"
"games where popularity greater than 200"

-- Rangos de valores (100% éxito)
"games where year between 2015 and 2020"
"games where rating between 3 and 5"
"games where popularity between 50 and 150"
```

### 📊 Consultas para Visualización (87% Éxito)

```sql
-- Distribuciones (perfectas para gráficos de barras)
"games group by year"
"games group by rating"
"games group by genres"

-- Análisis temporal (perfectas para líneas de tiempo)
"games where rating above 4 group by year"
"games where popularity above 100 group by year"
"games where year between 2018 and 2022 group by year"

-- Conteos y métricas
"count all games"
"count games where rating above 3"
"how many games are there"
```

### 🎮 Consultas por Géneros (83% Éxito)

```sql
-- Distribución de géneros
"games group by genres"
"games where rating above 4 group by genres"

-- Géneros específicos
"count games where genres contains Action"
"count games where genres contains RPG"
"count games where genres contains Strategy"
```

### ⚠️ Patrones a Evitar

```sql
❌ "show me 10 games"     → ✅ "games where rating above 3 limit 10"
❌ "top games"             → ✅ "games where rating above 4"
❌ "good games"            → ✅ "games where rating above 4"
❌ "popular games"         → ✅ "games where popularity above 100"
❌ "recent games"          → ✅ "games where year equals 2022"
```

---

## Formato de Entrada para Predicción

### Géneros (genres)
Lista de géneros del juego:
```json
["action", "rpg", "adventure", "shooter", "strategy", "simulation", "sports"]
```

### Plataformas (platforms)
Lista de plataformas objetivo:
```json
["pc", "playstation 4", "playstation 5", "xbox one", "nintendo switch"]
```

### Tags (tags)
Etiquetas descriptivas del juego:
```json
["multiplayer", "singleplayer", "co-op", "pvp", "open-world", "story-rich"]
```

### Duración Estimada (estimated_hours)
Horas de juego estimadas:
```json
30.0
```

### Año de Lanzamiento (release_year)
Año planeado de lanzamiento:
```json
2024
```

---

## Códigos de Respuesta

| Código | Descripción |
|--------|-------------|
| `200` | Consulta exitosa |
| `400` | Error en formato de entrada |
| `422` | Error de validación |
| `500` | Error interno del servidor |

---

## Validaciones y Restricciones

### Validación de Idioma y Sintaxis
- Solo consultas en **inglés** son soportadas para máxima precisión
- Detección automática de idioma con `langdetect`
- Mensaje de error amigable para otros idiomas
- **Sintaxis WHERE optimizada** recomendada para mejor rendimiento
- **Guías automáticas** hacia patrones exitosos

### Validación y Corrección SQL Ultra-Inteligente
- **Sanitización automática** de consultas SQL
- **Prevención de inyección SQL** con whitelist de operaciones
- **Corrección automática de patrones de fallo**:
  - GROUP BY problemático → Conversión automática a COUNT(*)
  - Condiciones imposibles → Eliminación de `rating > MAX(rating)`
  - LIKE mal formado → Limpieza de `'%RPG Group%'` → `'%RPG%'`
  - Columnas inventadas → Corrección de `ratings` → `rating`
  - Artefactos de parsing → Eliminación de condiciones inválidas
- **Traducción automática** de tabla: `games` → `games_optimized`
- **Corrección semántica** basada en contexto de la pregunta

### Límites de Consulta
- Máximo 1000 filas por consulta
- Timeout de 90 segundos para consultas complejas
- Validación de formato JSON
- Cache inteligente de 5 minutos para optimización

---

## Configuración y Variables de Entorno

### Variables Requeridas

```bash
DB_HOST=databaserawg.c72wwewsw0so.eu-north-1.rds.amazonaws.com
DB_NAME=DataRawg
DB_USER=postgres
DB_PASS=12345678-Hab
DB_PORT=5432
RAWG_API_KEY=9e9ab06256174d82a01dc73f02418ca2
```

### Variables Opcionales
```bash
API_BASE_URL=http://localhost:8000
LOG_LEVEL=INFO
```

---

## Instalación y Ejecución

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configuración de Variables
```bash
cp .env.example .env
# Editar .env con tus credenciales
```

### 3. Ejecución Local
```bash
# Opción A: Script directo
python api/api_v5/api_v5/run_api_v5.py

# Opción B: Uvicorn
uvicorn api.api_v5.api_v5.main_v5:app --reload --host 0.0.0.0 --port 8000
```

### 4. Verificación
```bash
curl http://localhost:8000/health
```

---

## Integración con Aplicaciones

### JavaScript/Frontend
```javascript
// Consulta de texto
const response = await fetch('http://localhost:8000/ask-text', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: 'top games by rating'})
});
const data = await response.json();

// Predicción de éxito
const prediction = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    genres: ['rpg', 'action'],
    platforms: ['pc'],
    tags: ['multiplayer'],
    estimated_hours: 25.0,
    release_year: 2024
  })
});
```

### Python
```python
import requests

# Consulta con visualización
response = requests.post('http://localhost:8000/ask-visual', 
  json={'question': 'games by genre'})
result = response.json()

# Predicción
prediction = requests.post('http://localhost:8000/predict', json={
  'genres': ['strategy'],
  'platforms': ['pc', 'mobile'],
  'tags': ['turn-based'],
  'estimated_hours': 15.0,
  'release_year': 2024
})
```

---

## Troubleshooting

### Errores Comunes

**Error 422 - Validation Error**
- Verificar formato JSON de entrada
- Asegurar que todos los campos requeridos estén presentes

**Error 500 - Database Connection**
- Verificar variables de entorno de base de datos
- Comprobar conectividad con `/test-model`

**Consulta en idioma no soportado**
- Traducir consulta al inglés
- Usar términos específicos de videojuegos

**Baja calidad en resultados SQL**
- Usar patrones WHERE optimizados documentados arriba
- Evitar términos abstractos como "good", "best", "top"
- Preferir sintaxis explícita: "games where rating above 4"

**Errores GROUP BY**
- El sistema corrige automáticamente la mayoría de casos
- Si persiste, usar consultas separadas en lugar de GROUP BY complejo

**Consultas sin resultados**
- Verificar rangos de valores (años: 1970-2024, rating: 0-5)
- Usar "games where" en lugar de "show me" o "list"

### Logs y Debugging

```bash
# Ver logs en tiempo real
tail -f logs/api.log

# Test de conectividad
curl http://51.20.113.231/test-model
curl http://51.20.113.231/test-fast-query

# Test de patrones optimizados
curl -X POST "http://51.20.113.231/ask-text" \
  -H "Content-Type: application/json" \
  -d '{"question": "games where year equals 2020"}'
```

---

## Despliegue en Producción

Para despliegue en AWS EC2, consultar la guía completa:
- **[EC2 Deployment Guide](EC2_DEPLOYMENT_GUIDE.md)**

### Características de Producción
- Nginx reverse proxy
- Systemd service
- UFW firewall
- Fail2Ban security
- Log rotation
- Health monitoring

---

## Soporte y Contacto

- **Documentación Interactiva**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Repositorio**: [Proyecto-RAWG](https://github.com/AlexGHerrera/Proyecto-RAWG)

---

*RAWG API v5 - Sistema de consultas NL→SQL con predicción ML para videojuegos*
