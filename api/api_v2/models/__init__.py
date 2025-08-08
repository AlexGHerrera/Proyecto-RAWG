"""
Paquete de modelos que expone funciones de predicción y ayudantes de consulta/visualización.
Este paquete abstrae el acceso a varios modelos utilizados por la API. Cada submódulo
puede importarse independientemente; importar ``models`` expone automáticamente
``predict``, ``ask_text`` y ``ask_visual`` para mayor comodidad.
"""

from . import predict  # noqa: F401
from . import ask_text  # noqa: F401
from . import ask_visual  # noqa: F401