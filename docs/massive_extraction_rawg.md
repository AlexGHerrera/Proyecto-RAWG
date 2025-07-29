# Extracción masiva RAWG (modo local)

## Objetivo

Construir un script ejecutable localmente que extraiga todos los videojuegos de la API de RAWG de forma masiva y segura, almacenando los datos como archivos `.json` en disco. El sistema está diseñado para ser reanudable y evitar duplicados si se interrumpe la ejecución.

---

## Lógica de funcionamiento

1. Pagina por la API de RAWG:
   - Endpoint: `https://api.rawg.io/api/games`
   - Utiliza los parámetros `?page` y `?page_size=40` (máximo permitido).
   - Extrae los datos en el orden predeterminado de la API.

2. Verifica si el archivo de la página ya existe:
   - Si existe, omite la descarga y muestra un mensaje de archivo ya existente.
   - Si no existe, realiza la petición a la API, guarda el archivo y muestra un mensaje de éxito.

3. Reanuda automáticamente:
   - El script puede detenerse en cualquier momento.
   - Al volver a ejecutarlo, continuará desde la última página descargada, evitando duplicados.

---

## Estructura de almacenamiento local

- Los archivos individuales se guardan en `data/raw/` con nombres tipo `games_page_123.json`.
- Los logs de actividad se almacenan en `logs/extraccion_rawg.log`.
- Cada archivo `.json` contiene una página completa de resultados, con las claves relevantes bajo `results`.

---

## Variables de entorno necesarias

```env
RAWG_API_KEY=...
```

Se recomienda el uso de un archivo `.env` y la librería `python-dotenv` para la gestión de variables.

---

## Dependencias requeridas

```txt
requests
tqdm
python-dotenv
```

Instalación recomendada:

```bash
pip install -r requirements.txt
```

---

## Beneficios del enfoque

- Robusto frente a interrupciones: permite reanudar la extracción sin perder progreso ni crear duplicados.
- Evita duplicados usando comprobaciones con `os.path.exists()`.
- Registro profesional mediante la librería `logging`.
- Escalable: los archivos generados pueden consolidarse, analizarse localmente o subirse a S3/RDS para su posterior procesamiento.

---

## Flujo de ejecución

```python
for page in range(1, total_pages + 1):
    if ya_existe_archivo(page):
        logging.info("Archivo de página %s ya existe, omitiendo", page)
    else:
        data = get_games_page(page)
        guardar_archivo(data, page)
        logging.info("Página %s guardada correctamente", page)
```

---

## Resultado esperado

- Una carpeta `data/raw/` con miles de archivos `games_page_X.json` numerados secuencialmente.
- Archivos listos para análisis local, consolidación, envío a S3 o carga masiva a RDS por otros procesos del pipeline.

