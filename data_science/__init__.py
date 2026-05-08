try:
    from importlib.metadata import version

    __version__ = version("data-science")
except ImportError:
    __version__ = "unknown"
