# 🚀 Lambda de Actualización Diaria RAWG → RDS

## 🌟 Objetivo

Diseñar una Lambda que se ejecute diariamente para mantener nuestra base de datos PostgreSQL (RDS) sincronizada con los juegos **nuevos y actualizados** en la API de RAWG.

---

## 📊 Lógica de funcionamiento

### ✅ 1. Consulta la API de RAWG

- Endpoint: `https://api.rawg.io/api/games`
- Parámetro clave: `?ordering=-updated`
  - Ordena los resultados por fecha de actualización descendente (los más recientes primero).

### ✅ 2. Recupera la fecha más reciente de actualización en RDS

```sql
SELECT MAX(updated) FROM games;
```

- Esto permite evitar procesar juegos que ya están actualizados.

### ✅ 3. Recorre las primeras N páginas (ej. 5-10) de RAWG

- Mientras `game["updated"] > max_updated_en_rds`
- Detiene el procesamiento cuando los juegos en RAWG ya no son más recientes.

### ✅ 4. Para cada juego:

- Si **no existe** en RDS → `INSERT`
- Si **existe** y `updated` es más reciente → `UPDATE`
- Si **existe** y no hay cambios → se ignora

---

## 🔢 Esquema de la tabla en PostgreSQL

```sql
CREATE TABLE IF NOT EXISTS games (
    id INT PRIMARY KEY,
    name TEXT,
    released DATE,
    rating FLOAT,
    updated TIMESTAMP,
    rawg_data JSONB
);
```

> ✉️ Se puede expandir el esquema con otros campos si se desea explotar datos específicos de RAWG.

---

## 🚧 Estructura de la Lambda

### 🌐 Origen de la ejecución

- Se programa vía EventBridge (frecuencia diaria).

### 📂 Ubicación esperada del código

```
data_pipeline/updater/lambda_update_rds.py
```

### 🔐 Variables de entorno necesarias

```env
RAWG_API_KEY=...
PG_HOST=...
PG_PORT=5432
PG_DATABASE=...
PG_USER=...
PG_PASSWORD=...
```

### 📃 Dependencias para Lambda

```txt
psycopg2-binary
requests
```

---

## ⚡️ Ventajas del enfoque

- ✅ Sincronización eficiente y precisa con RAWG.
- ✅ Mínimo volumen de datos por ejecución.
- ✅ Solo se procesan juegos nuevos o realmente modificados.
- ✅ Compatible con almacenamiento histórico en RDS y modelos futuros.

---

## 🔍 Detalles importantes

- No se debe usar `added` como referencia temporal: es una métrica de popularidad.
- `updated` es el campo correcto para detectar nuevas entradas y cambios.
- El `rawg_data` en formato `JSONB` actúa como copia de seguridad de toda la información original del juego.

---

## 🔄 SQL para actualización inteligente

```sql
INSERT INTO games (id, name, released, rating, updated, rawg_data)
VALUES (...)
ON CONFLICT (id) DO UPDATE
SET 
  name = EXCLUDED.name,
  released = EXCLUDED.released,
  rating = EXCLUDED.rating,
  updated = EXCLUDED.updated,
  rawg_data = EXCLUDED.rawg_data
WHERE EXCLUDED.updated > games.updated;
```

---

## 📄 Resultado esperado

Una base de datos en RDS que siempre contenga:

- Los últimos juegos añadidos a RAWG
- Las versiones más recientes de juegos ya existentes
- Un log estructurado de ejecuciones automáticas

