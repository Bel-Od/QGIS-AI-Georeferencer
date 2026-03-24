import sys
import io

# In some QGIS environments sys.stderr / sys.stdout can be None, which causes
# numpy (and other C extensions) to crash with AttributeError on import.
if sys.stderr is None:
    sys.stderr = io.StringIO()
if sys.stdout is None:
    sys.stdout = io.StringIO()


def classFactory(iface):
    from .plugin import AutoGeorefPlugin
    return AutoGeorefPlugin(iface)
