# Guía de Despliegue API v3 en AWS EC2

Esta guía te ayudará a desplegar la API v3 de predicción de videojuegos en una instancia EC2 de AWS.

## Requisitos Previos

- Cuenta de AWS activa
- Acceso a la consola de AWS
- Conocimientos básicos de Linux/Ubuntu
- Base de datos PostgreSQL accesible desde EC2 (RDS o instancia externa)

## Paso 1: Crear y Configurar la Instancia EC2

### 1.1 Lanzar Instancia EC2

1. **Accede a la Consola de AWS EC2**
   - Ve a AWS Console > EC2 > Launch Instance

2. **Configuración de la Instancia**
   - **Name**: `rawg-api-v3-server`
   - **AMI**: Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance Type**: `t3.medium` (recomendado para producción) o `t2.micro` (para pruebas)
   - **Key Pair**: Crea o selecciona una key pair existente
   - **Storage**: 20 GB gp3 (suficiente para la API y modelos)

3. **Configuración de Red**
   - **VPC**: Default VPC (o tu VPC personalizada)
   - **Subnet**: Public subnet
   - **Auto-assign Public IP**: Enable
   - **Security Group**: Crear nuevo con las siguientes reglas:
     - SSH (22): Tu IP
     - HTTP (80): 0.0.0.0/0
     - HTTPS (443): 0.0.0.0/0
     - Custom TCP (8000): 0.0.0.0/0 (puerto de la API)

### 1.2 Conectar a la Instancia

```bash
ssh -i "tu-key-pair.pem" ubuntu@tu-ec2-public-ip
```

## Paso 2: Configurar el Servidor

### 2.1 Actualizar Sistema

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Instalar Dependencias del Sistema

```bash
# Python y herramientas
sudo apt install -y python3 python3-pip python3-venv git nginx

# Dependencias para psycopg2
sudo apt install -y libpq-dev python3-dev

# Herramientas adicionales
sudo apt install -y htop curl unzip
```

### 2.3 Configurar Usuario de Aplicación

```bash
# Crear usuario para la aplicación
sudo useradd -m -s /bin/bash apiuser
sudo usermod -aG sudo apiuser

# Cambiar a usuario apiuser
sudo su - apiuser
```

## Paso 3: Desplegar la Aplicación

### 3.1 Subir Archivos de la API

Desde tu máquina local, sube los archivos:

```bash
# Comprimir la carpeta api_deploy
tar -czf api_deploy.tar.gz api_deploy/

# Subir a EC2
scp -i "tu-key-pair.pem" api_deploy.tar.gz ubuntu@tu-ec2-public-ip:~/

# En EC2, mover y extraer
sudo mv /home/ubuntu/api_deploy.tar.gz /home/apiuser/
sudo chown apiuser:apiuser /home/apiuser/api_deploy.tar.gz
sudo su - apiuser
tar -xzf api_deploy.tar.gz
```

### 3.2 Configurar Entorno Python

```bash
# Crear entorno virtual
cd api_deploy
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Configurar Variables de Entorno

```bash
# Copiar y editar archivo de configuración
cp .env.example .env
nano .env
```

Edita el archivo `.env` con tus credenciales de base de datos:

```env
# Database Configuration
DB_HOST=tu-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=rawg_database
DB_USER=tu_usuario
DB_PASS=tu_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=production

# Model Configuration
MODEL_CACHE_SIZE=100
QUERY_TIMEOUT=90
```

### 3.4 Probar la API Localmente

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar API en modo de prueba
cd api_v3
python run_api_v3.py
```

Verifica que la API responde en `http://tu-ec2-ip:8000/docs`

## Paso 4: Configurar Nginx como Proxy Reverso

### 4.1 Configurar Nginx

```bash
sudo nano /etc/nginx/sites-available/rawg-api
```

Contenido del archivo:

```nginx
server {
    listen 80;
    server_name tu-ec2-public-ip tu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Logs
    access_log /var/log/nginx/rawg-api.access.log;
    error_log /var/log/nginx/rawg-api.error.log;
}
```

### 4.2 Activar Configuración

```bash
# Crear enlace simbólico
sudo ln -s /etc/nginx/sites-available/rawg-api /etc/nginx/sites-enabled/

# Remover configuración por defecto
sudo rm /etc/nginx/sites-enabled/default

# Probar configuración
sudo nginx -t

# Reiniciar Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## Paso 5: Configurar Servicio Systemd

### 5.1 Crear Archivo de Servicio

```bash
sudo nano /etc/systemd/system/rawg-api.service
```

Contenido:

```ini
[Unit]
Description=RAWG API v3 Service
After=network.target

[Service]
Type=simple
User=apiuser
Group=apiuser
WorkingDirectory=/home/apiuser/api_deploy/api_v3
Environment=PATH=/home/apiuser/api_deploy/venv/bin
ExecStart=/home/apiuser/api_deploy/venv/bin/python run_api_v3.py
Restart=always
RestartSec=10

# Logs
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rawg-api

[Install]
WantedBy=multi-user.target
```

### 5.2 Activar Servicio

```bash
# Recargar systemd
sudo systemctl daemon-reload

# Habilitar servicio
sudo systemctl enable rawg-api

# Iniciar servicio
sudo systemctl start rawg-api

# Verificar estado
sudo systemctl status rawg-api
```

## Paso 6: Configurar SSL (Opcional pero Recomendado)

### 6.1 Instalar Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 6.2 Obtener Certificado SSL

```bash
# Solo si tienes un dominio
sudo certbot --nginx -d tu-dominio.com
```

## Paso 7: Monitoreo y Logs

### 7.1 Ver Logs de la API

```bash
# Logs del servicio
sudo journalctl -u rawg-api -f

# Logs de Nginx
sudo tail -f /var/log/nginx/rawg-api.access.log
sudo tail -f /var/log/nginx/rawg-api.error.log
```

### 7.2 Comandos Útiles

```bash
# Reiniciar API
sudo systemctl restart rawg-api

# Verificar estado de servicios
sudo systemctl status rawg-api
sudo systemctl status nginx

# Verificar uso de recursos
htop
df -h
free -h
```

## Paso 8: Pruebas de Funcionamiento

### 8.1 Probar Endpoints

```bash
# Health check
curl http://tu-ec2-ip/health

# Información del modelo
curl http://tu-ec2-ip/model/info

# Consulta de prueba
curl -X POST "http://tu-ec2-ip/ask-text" \
  -H "Content-Type: application/json" \
  -d '{"question": "games by platform"}'

# Consulta visual
curl -X POST "http://tu-ec2-ip/ask-visual" \
  -H "Content-Type: application/json" \
  -d '{"question": "top 10 games by rating"}'
```

### 8.2 Verificar Documentación

Accede a `http://tu-ec2-ip/docs` para ver la documentación interactiva de la API.

## Solución de Problemas Comunes

### Problema: API no responde
```bash
# Verificar si el servicio está corriendo
sudo systemctl status rawg-api

# Ver logs de errores
sudo journalctl -u rawg-api --no-pager -l
```

### Problema: Error de conexión a base de datos
```bash
# Verificar conectividad
telnet tu-rds-endpoint 5432

# Verificar variables de entorno
cat /home/apiuser/api_deploy/.env
```

### Problema: Nginx 502 Bad Gateway
```bash
# Verificar que la API esté corriendo en puerto 8000
sudo netstat -tlnp | grep 8000

# Verificar logs de Nginx
sudo tail -f /var/log/nginx/error.log
```

### Problema: Modelo no carga
```bash
# Verificar espacio en disco
df -h

# Verificar memoria
free -h

# Limpiar cache de Hugging Face si es necesario
rm -rf ~/.cache/huggingface/
```

## Costos Estimados AWS

- **t3.medium**: ~$30-35/mes
- **t2.micro** (Free Tier): Gratis por 12 meses
- **Storage (20GB)**: ~$2/mes
- **Data Transfer**: Variable según uso

## Seguridad Adicional

1. **Configurar Firewall UFW**:
```bash
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
```

2. **Actualizar regularmente**:
```bash
sudo apt update && sudo apt upgrade -y
```

3. **Monitorear logs regularmente**
4. **Configurar backups de la configuración**

## Contacto y Soporte

Para problemas específicos del despliegue, revisa los logs y verifica:
1. Conectividad de red
2. Estado de los servicios
3. Variables de entorno
4. Recursos del sistema (CPU, memoria, disco)

¡Tu API v3 debería estar funcionando correctamente en EC2!
