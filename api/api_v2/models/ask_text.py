"""
ask_text.py
============

Este módulo expone una interfaz simple para convertir preguntas en lenguaje natural
en consultas SQL. Delega la mayor parte del trabajo pesado a ``t5_model.py``, que
envuelve un modelo T5-small de HuggingFace. Una rutina de validación ligera asegura
que solo se acepten consultas de solo lectura contra el esquema RAWG conocido.

Ejemplo
-------

>>> from models import ask_text
>>> sql = ask_text.question_to_sql("¿Cuáles son los 10 juegos mejor valorados?")
>>> is_valid, message = ask_text.validar_sql_generada(sql)
"""

from __future__ import annotations

from typing import Tuple

# Reutiliza los ayudantes de conversión y validación definidos en t5_model. Se usa
# una importación absoluta simple aquí en lugar de una importación relativa porque
# el módulo ``t5_model.py`` reside en la raíz del proyecto en lugar de dentro
# del paquete ``models``. Python resuelve las importaciones de nivel superior usando
# ``sys.path`` que incluye la raíz del repositorio cuando se invoca vía
# ``python -m`` o cuando el paquete está en el PYTHONPATH.
import t5_model


def question_to_sql(user_question: str) -> str:
    """
    Genera una consulta SQL a partir de una pregunta en lenguaje natural.

    Parámetros
    ----------
    user_question : str
        La pregunta planteada por el usuario.

    Retorna
    -------
    str
        La consulta SQL generada por el modelo de lenguaje subyacente. Si la
        entrada está vacía se retorna una cadena vacía.
    """
    return t5_model.question_to_sql(user_question)


def validar_sql_generada(sql_code: str) -> Tuple[bool, str]:
    """
    Valida una declaración SQL usando las verificaciones básicas implementadas en
    ``t5_model.validate_sql``. Esta función se proporciona para compatibilidad
    hacia atrás con código anterior que hacía referencia a un validador con nombre en español.

    Parámetros
    ----------
    sql_code : str
        Cadena SQL para validar.

    Retorna
    -------
    tuple[bool, str]
        Un par de (es_valida, mensaje). ``es_valida`` será True si el
        SQL parece ser una consulta segura de solo lectura; de lo contrario False. ``mensaje``
        contiene la razón del error de validación o "Consulta válida".
    """
    return t5_model.validate_sql(sql_code)