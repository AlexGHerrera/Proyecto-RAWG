# Lambda de actualización diaria RAWG a RDS

## Objetivo

Diseñar una función Lambda que se ejecute diariamente y mantenga sincronizada la base de datos PostgreSQL (RDS) con los juegos nuevos y actualizados de la API de RAWG.

---

## Lógica de funcionamiento

1. Consulta la API de RAWG:
   - Endpoint principal: `https://api.rawg.io/api/games`
   - Utiliza el parámetro `?ordering=-updated` para obtener los juegos ordenados por fecha de actualización descendente.

2. Recupera la fecha más reciente de actualización registrada en RDS:

   ```sql
   SELECT MAX(updated) FROM games;
   ```
   - Así se evita procesar juegos que ya están actualizados en la base de datos.

3. Recorre las primeras N páginas de RAWG (por ejemplo, 5-10 páginas):
   - Mientras `game["updated"] > max_updated_en_rds`.
   - El proceso se detiene cuando los juegos en RAWG ya no son más recientes que el último en RDS.

4. Para cada juego recuperado:
   - Si no existe en RDS: se realiza un `INSERT`.
   - Si existe y el campo `updated` es más reciente: se realiza un `UPDATE`.
   - Si existe y no hay cambios: se ignora.

---

## Esquema de la tabla en PostgreSQL

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

*El esquema puede ampliarse con otros campos si se desea explotar más información de RAWG.*

---

## Estructura de la Lambda

- La ejecución se programa mediante EventBridge (frecuencia diaria).
- El código se ubica en: `data_pipeline/updater/lambda_update_rds.py`

### Variables de entorno necesarias

```env
RAWG_API_KEY=...
PG_HOST=...
PG_PORT=5432
PG_DATABASE=...
PG_USER=...
PG_PASSWORD=...
```

### Dependencias

```txt
psycopg2-binary
requests
```

---

## Beneficios del enfoque

- Sincronización eficiente y precisa con la API de RAWG.
- Mínimo volumen de datos procesados en cada ejecución.
- Solo se insertan o actualizan juegos realmente nuevos o modificados.
- Compatible con almacenamiento histórico y futuros modelos analíticos.

---

## Detalles importantes

- No se debe usar el campo `added` como referencia temporal, ya que mide popularidad y no actualización.
- El campo correcto para detectar cambios es `updated`.
- El campo `rawg_data` en formato JSONB permite conservar toda la información original del juego como respaldo.

---

## SQL para actualización inteligente

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

## Resultado esperado

- Una base de datos en RDS que contiene siempre los juegos más recientes y actualizados desde RAWG.
- Se mantienen tanto los nuevos lanzamientos como las versiones más recientes de juegos existentes.
- El proceso es automático y deja un registro estructurado de cada ejecución.

