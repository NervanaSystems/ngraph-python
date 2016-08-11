from geon.backends.graph.environment import get_current_environment

LayerNameKey = "****LAYERNAME****"


def node_with_layer_tag(node, tag_name):
    if get_current_environment() is not None:
        layer_name = get_current_environment().get_value(LayerNameKey, None)
        if layer_name is not None:
            node.tags.add((
                'LayerTag',
                layer_name,
                tag_name
            ))
    return node


def as_layer_input(node):
    return node_with_layer_tag(node, 'Input')


def as_layer_output(node):
    return node_with_layer_tag(node, 'Output')


def defined_in_layer(node):
    return node_with_layer_tag(node, 'DefinedIn')


def in_layer(f):
    def wrapper(self, *args, **kargs):
        get_current_environment().set_value(LayerNameKey, self.name)
        return f(self, *args, **kargs)
    return wrapper


def in_layer_config(config_func):
    @in_layer
    def wrapper(self, in_obj):
        in_obj = as_layer_input(in_obj)
        return as_layer_output(config_func(self, in_obj))
    return wrapper
