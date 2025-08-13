# RAWG API v5 - Guía de Uso y Documentación

[![API Version](https://img.shields.io/badge/API-v5.0.0-blue)](http://localhost:8000/docs)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](http://localhost:8000/health)
[![Model](https://img.shields.io/badge/Model-T5%20Fine--tuned-orange)](http://localhost:8000/model/info)

## Descripción General

La **RAWG API v5** es un sistema avanzado de consultas en lenguaje natural que convierte preguntas sobre videojuegos en consultas SQL optimizadas, genera visualizaciones automáticas y predice el éxito de videojuegos usando características de diseño.

### Características Principales

- **NL→SQL**: Modelo T5 fine-tuned para conversión de lenguaje natural a SQL
- **Vista Unificada**: Esquema optimizado `games_complete` con 500k+ juegos
- **Visualizaciones Automáticas**: Gráficos Plotly generados dinámicamente
- **Predicción ML**: Modelo de regresión para éxito de videojuegos
- **Validación Inteligente**: Corrección automática de SQL y validación de entrada
- **Seguridad**: Sanitización SQL y validación de idioma

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

## Ejemplos de Consultas Soportadas

### Consultas Básicas
- `"top 10 games by rating"`
- `"best RPG games"`
- `"games released in 2023"`
- `"most popular platforms"`

### Consultas Analíticas
- `"average rating by genre"`
- `"games with metacritic score > 90"`
- `"distribution of games by year"`
- `"platforms with most games"`

### Consultas Específicas
- `"action games on PC with rating > 4.0"`
- `"indie games released after 2020"`
- `"multiplayer games by platform"`
- `"games with playtime > 50 hours"`

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

### Validación de Idioma
- Solo consultas en **inglés** son soportadas
- Detección automática de idioma con `langdetect`
- Mensaje de error amigable para otros idiomas

### Validación SQL
- Sanitización automática de consultas SQL
- Prevención de inyección SQL
- Corrección inteligente de sintaxis

### Límites de Consulta
- Máximo 1000 filas por consulta
- Timeout de 30 segundos
- Validación de formato JSON

---

## Configuración y Variables de Entorno

### Variables Requeridas
```bash
DB_HOST=your-database-host
DB_NAME=your-database-name
DB_USER=your-username
DB_PASS=your-password
DB_PORT=5432
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

### Logs y Debugging
```bash
# Ver logs en tiempo real
tail -f logs/api.log

# Test de conectividad
curl http://localhost:8000/test-model
curl http://localhost:8000/test-fast-query
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
