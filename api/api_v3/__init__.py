"""
api_v3 - Sistema NL→SQL con T5 fine-tuned para RAWG
===================================================

Arquitectura unificada para generación de consultas SQL desde lenguaje natural
usando el modelo T5 fine-tuned y schema simplificado de PostgreSQL.

Módulos principales:
- sql_model_finetuned: Generación SQL con modelo fine-tuned
- ask_text: Endpoint de consultas de texto
- ask_visual: Endpoint de consultas con visualización
- main_v3: FastAPI principal

Características v3:
- Modelo fine-tuned: cssupport/t5-small-awesome-text-to-sql
- Schema RAWG simplificado (solo tablas esenciales)
- Prompt nativo del modelo fine-tuned
- Código compartido entre endpoints
- Validación y seguridad SQL mejoradas
- Visualizaciones automáticas optimizadas

Autor: API v3 Team
Fecha: 2025-01-09
"""

__version__ = "3.0.0"
__author__ = "API v3 Team"

# Funciones principales disponibles (importación lazy para evitar errores circulares)
__all__ = [
    "question_to_sql_finetuned",
    "get_model_info", 
    "test_model_connection"
]