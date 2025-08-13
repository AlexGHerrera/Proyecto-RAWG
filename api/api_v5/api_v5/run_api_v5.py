#!/usr/bin/env python3
"""
run_api_v5.py - Script de arranque para API v5
==============================================

Script principal para ejecutar la API v5 con importaciones relativas
correctas para despliegue en EC2.

Uso:
    python run_api_v5.py

Autor: API v5 Team
Fecha: 2025-01-12
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para importaciones
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    
    print("Iniciando API v5 para despliegue...")
    print(f"Directorio del proyecto: {project_root}")
    
    # Configuración para desarrollo con reload automático
    uvicorn.run(
        "main_v5:app",  # Importación relativa para el paquete de despliegue
        host="0.0.0.0",
        port=8000,
        reload=True,  # Activar reload para desarrollo
        log_level="info"
    )
