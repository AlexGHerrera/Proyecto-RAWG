
# 🧠 Configuración de AWS Lambda para ingesta y actualización de videojuegos RAWG

Este documento explica paso a paso cómo configurar una función AWS Lambda que:
- Se activa al subir un archivo JSON al bucket S3.
- Limpia y transforma los datos del archivo.
- Inserta o actualiza videojuegos en una base de datos PostgreSQL (AWS RDS).

---

## 🔧 Requisitos previos

- Un bucket S3 donde se subirán los archivos `.json` con videojuegos desde la API de RAWG.
- Una base de datos PostgreSQL en RDS, accesible desde la Lambda (en la misma VPC).
- Credenciales de acceso configuradas (usando IAM Role).

---

## 1. 📥 Crear la función Lambda

1. Ve a **AWS Lambda > Create function**
2. Elige: `Author from scratch`
3. Nombre sugerido: `rawg-json-ingestor`
4. Runtime: `Python 3.11`
5. Permissions:
   - Elige o crea un rol con:
     - `AmazonS3ReadOnlyAccess`
     - Permiso de acceso a tu RDS si es necesario (`AWSLambdaVPCAccessExecutionRole`)
   - Alternativamente, crea uno personalizado

---

## 2. 🧠 Variables de entorno

En la pestaña **Configuration > Environment variables**, añade:

| Key           | Value                       |
|---------------|-----------------------------|
| `DB_HOST`     | Host de tu RDS              |
| `DB_NAME`     | Nombre de la base de datos  |
| `DB_USER`     | Usuario                     |
| `DB_PASS`     | Contraseña                  |
| `DB_PORT`     | 5432 (por defecto)          |

---

## 3. 🛜 Conexión a RDS

En la pestaña **Configuration > VPC**:

1. Asocia la función a la misma **VPC/Subnet** donde está tu RDS.
2. Añade las **subnets privadas** (con NAT gateway o salida a internet).
3. Asocia los **Security Groups** que permitan salida al puerto `5432` (PostgreSQL).

---

## 4. ⚡ Crear el Trigger de S3

1. Ve a la pestaña **Configuration > Triggers**
2. Añade un trigger de tipo **S3**
3. Selecciona tu bucket
4. Evento: `PUT` (o `ObjectCreated`)
5. Filtra por prefijo o sufijo si lo deseas, por ejemplo: `rawg/` y `.json`

---

## 5. 📦 Subir el código de la Lambda

Puedes hacerlo de dos formas:

### Opción 1: Editor en línea
- Copia y pega el código Python directamente en la consola de Lambda.

### Opción 2: Subir archivo ZIP
- Incluye `psycopg2-binary` en un entorno local y empáquetalo junto con el script.
- O usa una Lambda Layer con `psycopg2`.

---

## 6. ✅ Validar funcionamiento

- Sube un archivo `.json` al bucket configurado.
- Verifica en **CloudWatch Logs** la ejecución paso a paso.
- Comprueba que los datos han sido insertados o actualizados en la base de datos.

---

## 🛑 Notas de seguridad

- Nunca subas contraseñas directamente al código.
- Usa `Secrets Manager` si quieres gestionar contraseñas de forma más segura.
- Limita los permisos del rol Lambda al bucket y recursos estrictamente necesarios.

---