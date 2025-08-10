# API v3 RAWG - Paquete de Despliegue

Este paquete contiene todos los archivos necesarios para desplegar la API v3 de predicción de videojuegos en AWS EC2.

## Contenido del Paquete

```
api_deploy/
├── api_v3/                     # Código fuente de la API
│   ├── main_v3.py             # Aplicación FastAPI principal
│   ├── run_api_v3.py          # Script de inicio
│   ├── models/                # Módulos de la API
│   │   ├── sql_model_finetuned.py  # Modelo T5 y lógica SQL
│   │   ├── ask_text.py        # Endpoint de consultas de texto
│   │   ├── ask_visual.py      # Endpoint de visualizaciones
│   │   ├── predict.py         # Endpoint de predicciones
│   │   └── utility_endpoints.py    # Endpoints de utilidad
│   └── __init__.py
├── requirements.txt           # Dependencias Python
├── .env.example              # Plantilla de variables de entorno
├── start_api.sh              # Script de inicio rápido
├── EC2_DEPLOYMENT_GUIDE.md   # Guía completa de despliegue
└── README.md                 # Este archivo
```

## Inicio Rápido

### 1. Configurar Variables de Entorno
```bash
cp .env.example .env
nano .env  # Editar con tus credenciales de BD
```

### 2. Iniciar API (Desarrollo)
```bash
./start_api.sh
```

### 3. Verificar Funcionamiento
- API: http://localhost:8000
- Documentación: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Endpoints Principales

- **POST /ask-text**: Consultas de texto en lenguaje natural
- **POST /ask-visual**: Consultas con visualizaciones
- **POST /predict**: Predicciones de éxito de videojuegos
- **GET /health**: Estado de la API
- **GET /model/info**: Información del modelo

## Variables de Entorno Requeridas

```env
# Base de Datos
DB_HOST=tu-host-postgresql
DB_PORT=5432
DB_NAME=rawg_database
DB_USER=tu_usuario
DB_PASS=tu_password

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Dependencias Principales

- **FastAPI**: Framework web
- **transformers**: Modelo T5 para SQL
- **psycopg2**: Conexión PostgreSQL
- **plotly**: Visualizaciones
- **langdetect**: Detección de idioma

## Características de la API v3

### ✅ Funcionalidades Implementadas
- Generación dinámica de SQL desde lenguaje natural
- Validación automática de consultas SQL
- Detección automática de idioma (solo inglés)
- Filtrado de consultas no relacionadas con videojuegos
- Visualizaciones automáticas con Plotly
- Predicciones de éxito de videojuegos
- Cache de consultas para mejor rendimiento
- Mensajes de error explicativos y user-friendly

### 🎯 Casos de Uso Soportados
- "games by platform" → Gráfico de juegos por plataforma
- "top 10 games by rating" → Lista de mejores juegos
- "What is the rating of Cyberpunk 2077?" → Rating específico
- Predicciones de éxito basadas en características del juego

### 🚫 Limitaciones Conocidas
- Solo acepta consultas en inglés
- Optimizado para base de datos RAWG específica
- El modelo T5 puede fallar en consultas muy complejas
- Requiere conexión activa a PostgreSQL

## Despliegue en Producción

Para despliegue completo en AWS EC2, sigue la guía detallada:
📖 **[EC2_DEPLOYMENT_GUIDE.md](EC2_DEPLOYMENT_GUIDE.md)**

## Solución de Problemas

### Error de Conexión a BD
```bash
# Verificar conectividad
telnet $DB_HOST $DB_PORT
```

### Modelo no Carga
```bash
# Verificar espacio y memoria
df -h && free -h
```

### API no Responde
```bash
# Verificar logs
tail -f logs/api.log  # Si existe
```

## Estructura de Respuestas

### Consulta de Texto Exitosa
```json
{
  "sql": "SELECT platform_name, COUNT(*) FROM games_view GROUP BY platform_name",
  "data": [...],
  "metadata": {
    "model": "cssupport/t5-small-awesome-text-to-sql",
    "rows_returned": 15,
    "elapsed_s": 1.234
  }
}
```

### Error de Validación
```json
{
  "sql": null,
  "data": [],
  "error": "This API only accepts questions in English...",
  "metadata": {
    "error_type": "non_english_query",
    "elapsed_s": 0.123
  }
}
```

## Rendimiento Esperado

- **Instancia recomendada**: t3.medium (2 vCPU, 4GB RAM)
- **Tiempo de respuesta**: 1-5 segundos por consulta
- **Throughput**: ~10-20 consultas concurrentes
- **Uso de memoria**: ~1-2GB (incluyendo modelo T5)

## Contacto y Soporte

Para problemas técnicos:
1. Revisar logs de la aplicación
2. Verificar conectividad de red y BD
3. Comprobar variables de entorno
4. Consultar la guía de despliegue completa

---

**Versión**: API v3  
**Modelo**: cssupport/t5-small-awesome-text-to-sql  
**Última actualización**: $(date)  
**Estado**: ✅ Listo para producción
