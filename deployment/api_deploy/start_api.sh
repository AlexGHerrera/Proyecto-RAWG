#!/bin/bash

# Script para iniciar la API v3 en producción
# Uso: ./start_api.sh

set -e

echo "🚀 Iniciando API v3 RAWG..."

# Verificar que estamos en el directorio correcto
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: No se encuentra requirements.txt. Ejecuta desde la carpeta api_deploy"
    exit 1
fi

# Activar entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar/actualizar dependencias
echo "📥 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Verificar archivo .env
if [ ! -f ".env" ]; then
    echo "⚠️  Advertencia: No se encuentra archivo .env"
    echo "   Copia .env.example a .env y configura tus variables"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "   Se ha creado .env desde .env.example"
    fi
fi

# Cambiar al directorio de la API
cd api_v3

# Verificar conectividad de base de datos (opcional)
echo "🔍 Verificando configuración..."
python -c "
import os
from dotenv import load_dotenv
load_dotenv('../.env')
required_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASS']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'❌ Variables faltantes: {missing}')
    exit(1)
else:
    print('✅ Variables de entorno configuradas correctamente')
"

# Iniciar la API
echo "🎯 Iniciando API en puerto 8000..."
echo "📖 Documentación disponible en: http://localhost:8000/docs"
echo "🛑 Presiona Ctrl+C para detener"

python run_api_v3.py
