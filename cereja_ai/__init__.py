from cereja.utils import get_version_pep440_compliant

# TODO: skip on build. Solve this
try:
    from . import vision
    from . import losses
    from . import metrics
except:
    pass

VERSION = "0.0.5.final.0"
__version__ = get_version_pep440_compliant(VERSION)
