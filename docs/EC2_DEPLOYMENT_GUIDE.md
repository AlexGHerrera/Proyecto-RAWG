# Guía de Despliegue API v3 en AWS EC2

Esta guía te ayudará a desplegar la API v3 de predicción de videojuegos en una instancia EC2 de AWS usando el paquete ZIP optimizado.

## Requisitos Previos

- Cuenta de AWS activa
- Acceso a la consola de AWS
- Conocimientos básicos de Linux/Ubuntu
- Base de datos PostgreSQL accesible desde EC2 (RDS o instancia externa)
- Archivo `rawg-api-v3-deployment-YYYYMMDD.zip` (paquete de despliegue)

## Paso 1: Crear y Configurar la Instancia EC2

### 1.1 Lanzar Instancia EC2

1. **Accede a la Consola de AWS EC2**
   - Ve a AWS Console > EC2 > Launch Instance

2. **Configuración de la Instancia**
   - **Name**: `rawg-api-v3-server`
   - **AMI**: Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance Type**: `t3.medium` (recomendado para producción) o `t2.micro` (para pruebas)
   - **Key Pair**: Crea o selecciona una key pair existente
   - **Storage**: 25 GB gp3 (suficiente para la API, modelos y sistema)

3. **Configuración de Red**
   - **VPC**: Default VPC (o tu VPC personalizada)
   - **Subnet**: Public subnet
   - **Auto-assign Public IP**: Enable
   - **Security Group**: Crear nuevo con las siguientes reglas:
     - SSH (22): Tu IP específica (no 0.0.0.0/0 por seguridad)
     - HTTP (80): 0.0.0.0/0
     - HTTPS (443): 0.0.0.0/0
     - Custom TCP (8000): 0.0.0.0/0 (puerto de la API - temporal para pruebas)

### 1.2 Conectar a la Instancia

```bash
# Asegurar permisos correctos de la clave
chmod 400 tu-key-pair.pem

# Conectar a la instancia
ssh -i "tu-key-pair.pem" ubuntu@tu-ec2-public-ip
```

## Paso 2: Configurar el Servidor

### 2.1 Actualizar Sistema

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Instalar Dependencias del Sistema

```bash
# Python y herramientas esenciales
sudo apt install -y python3 python3-pip python3-venv git nginx unzip

# Dependencias para psycopg2 y compilación
sudo apt install -y libpq-dev python3-dev build-essential

# Herramientas de monitoreo y utilidades
sudo apt install -y htop curl tree
```

### 2.3 Configurar Usuario de Aplicación

```bash
# Crear usuario dedicado para la aplicación
sudo useradd -m -s /bin/bash apiuser

# Cambiar a usuario apiuser
sudo su - apiuser
```

## Paso 3: Desplegar la Aplicación

### 3.1 Transferir Paquete ZIP a EC2

Desde tu máquina local:

```bash
# Transferir el paquete ZIP a EC2
scp -i "tu-key-pair.pem" rawg-api-v3-deployment-20250810.zip ubuntu@tu-ec2-public-ip:~/

# Verificar transferencia exitosa
ssh -i "tu-key-pair.pem" ubuntu@tu-ec2-public-ip "ls -lh ~/rawg-api-v3-deployment-*.zip"
```

### 3.2 Descomprimir y Configurar

En la instancia EC2:

```bash
# Mover archivo al directorio del usuario apiuser
sudo mv /home/ubuntu/rawg-api-v3-deployment-*.zip /home/apiuser/
sudo chown apiuser:apiuser /home/apiuser/rawg-api-v3-deployment-*.zip

# Cambiar a usuario apiuser
sudo su - apiuser
cd ~

# Descomprimir el paquete
unzip rawg-api-v3-deployment-*.zip

# Verificar contenido
ls -la api_deploy/
tree api_deploy/ -L 2
```

### 3.3 Configurar Entorno Python

```bash
# Navegar al directorio de la aplicación
cd api_deploy

# Ejecutar script de inicio automático
chmod +x start_api.sh
./start_api.sh
```

**Nota**: El script `start_api.sh` automáticamente:

- Crea el entorno virtual
- Instala todas las dependencias desde `requirements.txt`
- Configura las variables de entorno

### 3.4 Configurar Variables de Entorno

```bash
# Copiar y editar archivo de configuración
cp .env.example .env
nano .env
```

Edita el archivo `.env` con tus credenciales reales:

```env
# Database Configuration
DB_HOST=tu-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=rawg_database
DB_USER=tu_usuario
DB_PASS=tu_password

# RAWG API Key (obtener de https://rawg.io/apidocs)
RAWG_API_KEY=tu_rawg_api_key

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=production

# Model Configuration
MODEL_CACHE_SIZE=100
QUERY_TIMEOUT=90

# S3 Configuration (opcional)
S3_BUCKET_NAME=tu-bucket-name
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
```

### 3.5 Probar la API Localmente

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar API en modo de prueba
cd api_v3
python run_api_v3.py
```

Verifica que la API responde correctamente:

- Documentación: `http://tu-ec2-ip:8000/docs`
- Health check: `http://tu-ec2-ip:8000/health`
- Información del modelo: `http://tu-ec2-ip:8000/model/info`

**Importante**: Detén la API con `Ctrl+C` antes de continuar con la configuración del servicio systemd.

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

# Consulta de texto
curl -X POST "http://tu-ec2-ip/ask-text" \
  -H "Content-Type: application/json" \
  -d '{"question": "games by platform"}'

# Consulta visual
curl -X POST "http://tu-ec2-ip/ask-visual" \
  -H "Content-Type: application/json" \
  -d '{"question": "top 10 games by rating"}'

# Predicción de éxito
curl -X POST "http://tu-ec2-ip/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "genres": "action+adventure",
    "platforms": "pc+playstation 4",
    "tags": "multiplayer, open world",
    "estimated_hours": 30
  }'

# Visualización HTML directa
curl -X POST "http://tu-ec2-ip/ask-visual-html" \
  -H "Content-Type: application/json" \
  -d '{"question": "games by genre"}' > test_chart.html
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

## Paso 9: Optimización y Mantenimiento

### 9.1 Configurar Rotación de Logs

```bash
# Crear configuración de logrotate
sudo nano /etc/logrotate.d/rawg-api
```

Contenido:

```textr/log/nginx/rawg-api.*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 www-data www-data
    postrotate
        systemctl reload nginx
    endscript
}
```

### 9.2 Configurar Monitoreo de Recursos

```bash
# Instalar herramientas de monitoreo
sudo apt install -y iotop nethogs

# Script de monitoreo básico
cat > ~/monitor_api.sh << 'EOF'
#!/bin/bash
echo "=== API Status ==="
sudo systemctl status rawg-api --no-pager -l
echo "\n=== Resource Usage ==="
free -h
df -h /
echo "\n=== Network Connections ==="
sudo netstat -tlnp | grep :8000
EOF

chmod +x ~/monitor_api.sh
```

## Costos Estimados AWS

- **t3.medium**: ~$30-35/mes (recomendado para producción)
- **t2.micro** (Free Tier): Gratis por 12 meses (solo para pruebas)
- **Storage (25GB gp3)**: ~$2.5/mes
- **Data Transfer**: Variable según uso (~$0.09/GB salida)
- **RDS PostgreSQL** (si aplica): Desde $15/mes

## Seguridad y Mejores Prácticas

### 1. Configurar Firewall UFW

```bash
# Habilitar firewall
sudo ufw --force enable

# Permitir solo puertos necesarios
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'

# Denegar acceso directo al puerto 8000 desde internet
sudo ufw deny 8000

# Verificar reglas
sudo ufw status verbose
```

### 2. Configurar Fail2Ban

```bash
# Instalar fail2ban
sudo apt install -y fail2ban

# Crear configuración personalizada
sudo nano /etc/fail2ban/jail.local
```

Contenido:

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
```

### 3. Actualización y Mantenimiento

```bash
# Script de actualización automática
cat > ~/update_system.sh << 'EOF'
#!/bin/bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo systemctl restart rawg-api
sudo systemctl restart nginx
echo "System updated: $(date)" >> ~/update.log
EOF

chmod +x ~/update_system.sh

# Configurar cron para actualizaciones semanales
(crontab -l 2>/dev/null; echo "0 2 * * 0 /home/ubuntu/update_system.sh") | crontab -
```

### 4. Backup de Configuración

```bash
# Script de backup
cat > ~/backup_config.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/ubuntu/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configuraciones importantes
cp /home/apiuser/api_deploy/.env $BACKUP_DIR/
cp /etc/nginx/sites-available/rawg-api $BACKUP_DIR/
cp /etc/systemd/system/rawg-api.service $BACKUP_DIR/

echo "Backup completed: $(date)" >> ~/backup.log
EOF

chmod +x ~/backup_config.sh

# Ejecutar backup semanal
(crontab -l 2>/dev/null; echo "0 3 * * 0 /home/ubuntu/backup_config.sh") | crontab -
```

## Paso 10: Verificación Final y Checklist

### 10.1 Checklist de Despliegue

- [ ] Instancia EC2 creada y configurada
- [ ] Dependencias del sistema instaladas
- [ ] Usuario `apiuser` creado
- [ ] Paquete ZIP transferido y descomprimido
- [ ] Script `start_api.sh` ejecutado exitosamente
- [ ] Variables de entorno configuradas en `.env`
- [ ] API probada localmente en puerto 8000
- [ ] Nginx configurado como proxy reverso
- [ ] Servicio systemd creado y habilitado
- [ ] Firewall UFW configurado
- [ ] Fail2Ban instalado y configurado
- [ ] SSL configurado (opcional)
- [ ] Logs y monitoreo configurados
- [ ] Scripts de backup y actualización creados
- [ ] Todos los endpoints probados exitosamente

### 10.2 Comandos de Verificación Final

```bash
# Verificar estado de todos los servicios
sudo systemctl status rawg-api nginx fail2ban

# Verificar puertos abiertos
sudo netstat -tlnp | grep -E ':(80|443|8000)'

# Verificar firewall
sudo ufw status verbose

# Verificar logs recientes
sudo journalctl -u rawg-api --since "1 hour ago" --no-pager

# Verificar espacio en disco
df -h

# Verificar memoria
free -h

# Probar conectividad de base de datos
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT version();"
```

### 10.3 URLs de Verificación

Una vez completado el despliegue, verifica estos endpoints:

- **Documentación API**: `http://tu-ec2-ip/docs`
- **Health Check**: `http://tu-ec2-ip/health`
- **Información del Modelo**: `http://tu-ec2-ip/model/info`
- **Prueba de Consulta**: `http://tu-ec2-ip/ask-text` (POST)
- **Prueba de Visualización**: `http://tu-ec2-ip/ask-visual-html` (POST)
- **Prueba de Predicción**: `http://tu-ec2-ip/predict` (POST)

## Troubleshooting Avanzado

### Logs Importantes

```bash
# Logs de la aplicación
sudo journalctl -u rawg-api -f

# Logs de Nginx
sudo tail -f /var/log/nginx/rawg-api.access.log
sudo tail -f /var/log/nginx/rawg-api.error.log

# Logs del sistema
sudo tail -f /var/log/syslog

# Logs de autenticación
sudo tail -f /var/log/auth.log
```

### Comandos de Diagnóstico

```bash
# Verificar procesos Python
ps aux | grep python

# Verificar conexiones de red
sudo ss -tlnp | grep :8000

# Verificar uso de memoria por proceso
sudo ps aux --sort=-%mem | head -10

# Verificar uso de CPU
top -p $(pgrep -d',' python)

# Verificar archivos abiertos por la aplicación
sudo lsof -p $(pgrep -f "run_api_v3.py")
```

## Contacto y Soporte

Para soporte técnico o consultas sobre el despliegue:

- **Documentación del Proyecto**: Revisar README.md en el repositorio
- **Logs de Error**: Siempre incluir logs relevantes al reportar problemas  
- **Información del Sistema**: Incluir versión de Ubuntu, recursos de EC2, etc.

---

**¡Despliegue Completado!** Tu API v3 de predicción de videojuegos RAWG está ahora funcionando en producción en AWS EC2.

Para problemas específicos del despliegue, revisa los logs y verifica:
1. Conectividad de red
2. Estado de los servicios
3. Variables de entorno
4. Recursos del sistema (CPU, memoria, disco)

¡Tu API v3 debería estar funcionando correctamente en EC2!
