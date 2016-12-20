# TODO: move transformer_name to other file? or move to here to keep all things added for flex in same place
from ngraph.testing.error_check import transformer_name

# FLEX TODO: this is not exactly what I want, skips the tests instead of marking xfail
# TODO: keep this in its own flex specific file? so we remember this was added for flex
def xfail_transformer_type(transfrmer_name, reason=None, run=False):
    def decorator(test_func):
        if transformer_name() == transfrmer_name:
            return pytest.mark.xfail(reason=reason, run=run)
    return decorator
