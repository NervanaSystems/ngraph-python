try:
    import tensorflow as _  # noqa
except ImportError:
    raise ImportError("Tensorflow is not installed, yet it is required "
                      "to export Nervana Graph attributes to TensorBoard")
