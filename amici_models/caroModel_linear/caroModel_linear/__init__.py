"""AMICI-generated module for model caroModel_linear"""

import amici
from pathlib import Path

# Ensure we are binary-compatible, see #556
if '0.16.0' != amici.__version__:
    raise amici.AmiciVersionError(
        f'Cannot use model `caroModel_linear` in {Path(__file__).parent}, '
        'generated with amici==0.16.0, '
        f'together with amici=={amici.__version__} '
        'which is currently installed. To use this model, install '
        'amici==0.16.0 or re-import the model with the amici '
        'version currently installed.'
    )

from .caroModel_linear import *
from .caroModel_linear import getModel as get_model

__version__ = '0.1.0'
