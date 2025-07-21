{\rtf1\ansi\ansicpg1252\cocoartf2860
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 proyecto-rawg/\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  README.md\
\uc0\u9500 \u9472 \u9472  .gitignore\
\uc0\u9500 \u9472 \u9472  requirements.txt\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  docs/\
\uc0\u9474    \u9492 \u9472 \u9472  arquitectura_aws.png        # Diagramas, esquemas, documentaci\'f3n t\'e9cnica\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  infra/\
\uc0\u9474    \u9500 \u9472 \u9472  terraform/                  # (Opcional) IaC para AWS si usan Terraform\
\uc0\u9474    \u9492 \u9472 \u9472  config/                     # Configuraci\'f3n de acceso, variables, etc.\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  data_pipeline/\
\uc0\u9474    \u9500 \u9472 \u9472  rawg_extractor/\
\uc0\u9474    \u9474    \u9500 \u9472 \u9472  lambda_massive.py       # Extracci\'f3n masiva de juegos\
\uc0\u9474    \u9474    \u9500 \u9472 \u9472  lambda_daily.py         # Extracci\'f3n diaria de juegos\
\uc0\u9474    \u9474    \u9492 \u9472 \u9472  utils.py\
\uc0\u9474    \u9492 \u9472 \u9472  loader/\
\uc0\u9474        \u9492 \u9472 \u9472  lambda_loader.py        # Carga desde S3 a PostgreSQL\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  data/\
\uc0\u9474    \u9500 \u9472 \u9472  sample/                     # Muestras de JSONs extra\'eddos\
\uc0\u9474    \u9492 \u9472 \u9472  schema.sql                  # Estructura de la base de datos\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  model/\
\uc0\u9474    \u9500 \u9472 \u9472  train_model.py              # Entrenamiento del modelo de \'e9xito\
\uc0\u9474    \u9500 \u9472 \u9472  model.pkl                   # Modelo entrenado (o versi\'f3n m\'ednima)\
\uc0\u9474    \u9492 \u9472 \u9472  evaluate_model.ipynb        # M\'e9tricas de validaci\'f3n\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  api/\
\uc0\u9474    \u9500 \u9472 \u9472  main.py                     # Aplicaci\'f3n FastAPI\
\uc0\u9474    \u9500 \u9472 \u9472  routes/\
\uc0\u9474    \u9474    \u9500 \u9472 \u9472  predict.py              # Endpoint /predict\
\uc0\u9474    \u9474    \u9500 \u9472 \u9472  ask_text.py             # Endpoint /ask-text\
\uc0\u9474    \u9474    \u9492 \u9472 \u9472  ask_visual.py           # Endpoint /ask-visual\
\uc0\u9474    \u9492 \u9472 \u9472  utils/\
\uc0\u9474        \u9492 \u9472 \u9472  query_generator.py      # L\'f3gica NLP \u8594  SQL\
\uc0\u9474 \
\uc0\u9492 \u9472 \u9472  deployment/\
    \uc0\u9492 \u9472 \u9472  ec2_setup.sh                # Script para desplegar en EC2}