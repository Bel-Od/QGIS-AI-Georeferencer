def classFactory(iface):
    from .plugin import AutoGeorefPlugin
    return AutoGeorefPlugin(iface)
