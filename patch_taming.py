"""
===============================================================================
AUTO-PATCH PARA TAMING-TRANSFORMERS
===============================================================================

Este modulo aplica automaticamente el fix de compatibilidad torch._six a
taming-transformers SIN necesidad de modificar el repositorio clonado.

Se ejecuta ANTES de importar cualquier modulo de taming-transformers.
"""

import sys
import importlib.util

# Arreglar encoding del stdout para Windows (evita UnicodeEncodeError con caracteres como Unicode)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # Si falla el rewrap, seguir de todas formas


def patch_taming_transformers():
    """
    Parchea automaticamente torch._six en taming-transformers.

    Este fix previene el error:
    ModuleNotFoundError: No module named 'torch._six'

    PyTorch 1.9+ removio torch._six, pero taming-transformers aun lo usa.
    """
    # Crear modulo virtual torch._six si no existe
    if 'torch._six' not in sys.modules:
        try:
            import torch

            # Crear modulo virtual
            import types
            torch_six = types.ModuleType('torch._six')

            # Definir string_classes (usado por taming/data/utils.py)
            torch_six.string_classes = str

            # Inyectar en sys.modules
            sys.modules['torch._six'] = torch_six

            print("[OK] Patch automatico aplicado: torch._six creado")

        except ImportError:
            print("[AVISO] PyTorch no instalado, patch omitido")
    else:
        print("[OK] torch._six ya existe (PyTorch antiguo o ya parcheado)")


# Aplicar patch al importar este modulo
patch_taming_transformers()
