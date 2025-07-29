# Carga Masiva de Juegos RAWG a PostgreSQL — `massive_loader.py`

## Objetivo

Script diseñado para cargar de forma masiva los datos de videojuegos extraídos de la API de RAWG (almacenados en archivos JSON en S3) a una base de datos PostgreSQL, procesando eficientemente relaciones y metadatos asociados.

---

## Lógica de funcionamiento

1. **Configuración y entorno:**
   - Carga variables de entorno críticas para la conexión a la base de datos y S3 (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_PORT`, `S3_BUCKET`).
   - Inicializa el logger y los logs de actividad.

2. **Reanudación automática:**
   - El script guarda el progreso en `resume_state.txt` para poder continuar desde el último archivo procesado en caso de interrupción.

3. **Conexión y preparación:**
   - Se conecta a la base de datos PostgreSQL.
   - Inicializa un cache en memoria para evitar inserciones duplicadas de metadatos en la misma ejecución.

4. **Procesamiento de archivos JSON desde S3:**
   - Lista todos los archivos JSON bajo el prefijo configurado en el bucket S3.
   - Itera sobre cada archivo (paginado), lee y parsea su contenido.
   - Extrae la lista de juegos de la clave `results`.

5. **Carga masiva de juegos:**
   - Inserta o actualiza en bloque los registros principales de la tabla `games` usando `execute_values` y `ON CONFLICT DO UPDATE` para mantener los datos frescos.

6. **Procesamiento modular de relaciones y metadatos:**
   - Procesa metadatos y relaciones (ESRB, plataformas, géneros, tags, tiendas, screenshots, ratings, estados) agrupando los datos por lote para eficiencia.
   - Cada función modular se encarga de insertar/actualizar solo si es necesario, usando el cache para evitar duplicados.

7. **Control de errores y logs:**
   - Los errores se capturan y registran, pero no detienen el procesamiento del resto de archivos.
   - El progreso se guarda tras cada archivo procesado correctamente.

8. **Finalización:**
   - Al completar todos los archivos, registra el tiempo total de ejecución y finaliza.

---

## Variables de entorno necesarias

```env
DB_HOST=...
DB_NAME=...
DB_USER=...
DB_PASS=...
DB_PORT=5432
S3_BUCKET=...
```

---

## Dependencias requeridas

```txt
psycopg2-binary
boto3
dotenv
tqdm
```

Instalación recomendada:

```bash
pip install -r requirements.txt
```

---

## Beneficios del enfoque

- Permite cargar grandes volúmenes de datos de manera eficiente y segura.
- Es tolerante a fallos y reanudable.
- Procesa todas las relaciones y metadatos relevantes de forma modular y escalable.
- Registra el progreso y los errores para facilitar el seguimiento y debug.

---

## Flujo de ejecución resumido

```python
for i, key in enumerate(archivos_json_s3):
    payload = leer_json_s3(key)
    all_games = payload.get("results", [])
    # Inserción masiva de juegos
    execute_values(cur, "INSERT INTO games ... ON CONFLICT DO UPDATE ...", valores)
    # Procesamiento modular de relaciones
    procesar_esrb(...)
    procesar_platforms_batch(...)
    procesar_tags(...)
    # ...etc
    guardar_progreso(i)
```

---

## Resultado esperado

- Todos los juegos y sus relaciones principales de los archivos JSON quedan correctamente cargados y actualizados en la base de datos PostgreSQL.
- El proceso puede ser reanudado si se interrumpe, y es robusto frente a errores en archivos individuales.
- Los logs y el archivo de estado permiten auditar y continuar la operación fácilmente.
