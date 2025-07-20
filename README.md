# Propuesta de Proyecto: Pipeline y API de Análisis de Videojuegos con RAWG

## 👥 Roles y Responsabilidades

### 🏗️ Data Architect
- **Responsable del Diseño en AWS:** Definir y configurar la arquitectura en la nube, incluyendo el bucket S3 para el Data Lake, la base de datos PostgreSQL en RDS y los permisos necesarios para los servicios.

### ⚙️ Data Engineer
- **Responsable del Pipeline de Datos:** Implementar las funciones AWS Lambda para la extracción masiva (con paginación) y diaria (con filtro de fecha) de datos de la API de RAWG, y para la carga de estos datos desde S3 hacia la base de datos en RDS.

### 🔬 Data Scientist
- **Responsable del Análisis y Modelado:**
  - Analizar los datos en PostgreSQL para definir una métrica de "éxito" para un videojuego.
  - Entrenar un modelo de clasificación para predecir dicho éxito.
  - Diseñar la lógica para los endpoints de Q&A, seleccionando los modelos apropiados de Hugging Face.

### 🚀 ML Engineer
- **Responsable del Despliegue y la API:**
  - Envolver el modelo predictivo y la lógica de los endpoints en una API robusta utilizando FastAPI.
  - Desplegar la aplicación FastAPI en una instancia de AWS EC2, asegurando que todos los endpoints sean funcionales.

---

## 📝 Fases del Proyecto

### **Fase 01: Infraestructura y Pipeline de Datos**
**Objetivo:** Construir un sistema automatizado que extraiga datos de la API de RAWG y los almacene de forma estructurada en una base de datos en la nube.

**Tareas Clave:**
1.  **Extracción de Datos (Data Engineer, Data Architect):**
    - Obtener una API Key de RAWG.
    - Crear una **AWS Lambda** para la extracción masiva inicial de todos los videojuegos y guardarlos en un **bucket S3**.
    - Configurar la Lambda con **EventBridge** para que se ejecute diariamente y obtenga solo los juegos actualizados recientemente.

2.  **Procesamiento y Carga (Data Engineer, Data Architect):**
    - Desarrollar una segunda **AWS Lambda** que se active mediante un **trigger de S3** al recibir nuevos datos.
    - Esta función procesará los ficheros JSON y los cargará de forma estructurada en la base de datos **PostgreSQL en AWS RDS**.

**Entregables de esta fase:**
- Infraestructura en AWS (S3, RDS) configurada.
- Pipelines de datos automáticos (masivo y diario) funcionando.
- Base de datos poblada y actualizada con los datos de RAWG.

### **Fase 02: Modelado, API y Despliegue**
**Objetivo:** Desarrollar una API multifuncional que prediga el éxito de un videojuego y permita hacer consultas complejas.

**Tareas Clave:**
1.  **Desarrollo del Modelo (Data Scientist):**
    - Utilizar los datos de PostgreSQL para entrenar un modelo de Machine Learning (ej. con XGBoost) que prediga si un videojuego será un "éxito".

2.  **Desarrollo de la API (ML Engineer, Data Scientist):**
    - Crear una aplicación con **FastAPI** que incluya los siguientes endpoints:
      - `/predict`: Recibe los datos de un videojuego y devuelve la predicción del modelo.
      - `/ask-text`: Recibe una pregunta en texto (ej. "¿Desarrollador con la mejor puntuación?"), la convierte a SQL usando un modelo de **Hugging Face**, consulta la base de datos y devuelve una respuesta en texto.
      - `/ask-visual`: Recibe una pregunta orientada a visualización (ej. "Top 10 géneros por número de juegos"), consulta la base de datos y devuelve un gráfico generado con **Matplotlib/Seaborn**.

3.  **Despliegue (ML Engineer):**
    - Desplegar la aplicación FastAPI completa en una instancia **AWS EC2** para que sea accesible públicamente.

**Entregables de esta fase:**
- Un modelo de clasificación entrenado.
- API funcional desplegada en EC2 con los tres endpoints implementados.
- Documentación de la API (generada por FastAPI).