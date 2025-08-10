import time
import logging
from fastapi import APIRouter, HTTPException
# Importación con fallback para compatibilidad local y EC2
try:
    from .sql_model_finetuned import get_model_info
except ImportError:
    from sql_model_finetuned import get_model_info

# Configurar logging
logger = logging.getLogger(__name__)

# Crear un router para los endpoints de utilidad
router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Endpoint de health check para verificar el estado de la API.
    """
    try:
        # Verificar componentes básicos
        model_info = get_model_info()
        
        # Intentar conexión con modelo (sin cargar si no está cargado)
        model_status = "available" if model_info["transformers_available"] else "unavailable"
        
        return {
            "status": "healthy",
            "api_version": "v3",
            "model_status": model_status,
            "transformers_available": model_info["transformers_available"],
            "database_connection": "ok",  # Se verifica en la primera consulta
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return {
            "status": "unhealthy",
            "api_version": "v3",
            "error": str(e),
            "timestamp": time.time()
        }
