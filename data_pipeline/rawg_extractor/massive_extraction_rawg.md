# 🤖 Extracción Masiva RAWG (modo local)

## 🌟 Objetivo

Construir un script ejecutable localmente que extrae **todos los videojuegos** de la API de RAWG, de forma masiva y segura, para almacenamiento inicial en disco como archivos `.json`. El sistema está diseñado para ser **reanudable**, evitando duplicados si se interrumpe.

---

## 📊 Lógica de funcionamiento

### ✅ 1. Pagina por la API de RAWG

- Endpoint: `https://api.rawg.io/api/games`
- Usa `?page` y `?page_size=40` (máximo permitido).
- Extrae los datos en orden predeterminado.

### ✅ 2. Verifica si el archivo de la página ya existe

- Si existe: lo omite con mensaje `[YA EXISTE]`.
- Si no existe: llama a la API, guarda el archivo y muestra `[GUARDADA]`.

### ✅ 3. Se reanuda automáticamente

- Puedes detener el script en cualquier momento.
- Al volver a ejecutarlo, continuará desde donde se quedó.

---

## 🔍 Estructura de almacenamiento local

### 📁 Directorios utilizados

- `data/raw/` → almacenamiento de páginas individuales (`games_page_123.json`)
- `logs/` → registro de actividad en `extraccion_rawg.log`

### 📔 Formato de los archivos

- Cada página se guarda como un archivo `.json` legible y completo.
- Las claves relevantes están dentro de `results`.

---

## 🔐 Variables de entorno necesarias

```env
RAWG_API_KEY=...
```

Opcionalmente puedes usar `.env` con `python-dotenv`.

---

## 📃 Dependencias requeridas

```txt
requests
tqdm
python-dotenv
```

Instalar con:

```bash
pip install -r requirements.txt
```

---

## ⚡️ Ventajas del enfoque

- ✅ Robusto frente a cortes de ejecución.
- ✅ Evita duplicados con `os.path.exists()`.
- ✅ Registro profesional con `logging`.
- ✅ Escalable: puedes consolidar más tarde en un único JSON, CSV o cargar a S3/RDS.

---

## 🔄 Flujo de ejecución

```python
for page in range(1, total_pages + 1):
    if ya_existe_archivo(page):
        logging.info("[YA EXISTE] Página {page} omitida")
    else:
        data = get_games_page(page)
        guardar_archivo(data, page)
        logging.info("[GUARDADA] Página {page} correctamente")
```

---

## 📄 Resultado esperado

Una carpeta `data/raw/` con miles de archivos `games_page_X.json` numerados secuencialmente, listos para:

- Análisis local
- Consolidación
- Enviarlos a S3 para procesamiento
- Carga masiva a RDS por otra Lambda posterior

