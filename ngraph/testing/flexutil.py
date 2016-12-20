from ngraph.testing.error_check import transformer_name

# not sure why this doesn't work
def xfail_if_flexgpu_old(reason=None, run=False):
    def decorator(test_func):
        if transformer_name() == 'flexgpu':
            return pytest.mark.xfail(reason=reason, run=run)(test_func)
        else:
            return test_func
    return decorator

def xfail_if_flexgpu(reason=None, run=False):
    def decorator(test_func):
        # for test_conv, this will print numpy, False 4 times for first 4 undecorated tests
        #print transformer_name()
        #print transformer_name() == 'flexgpu'
        # FLEX TODO: this is not exactly what I want, skips the tests instead of marking xfail
        if transformer_name() == 'flexgpu':
            return pytest.mark.xfail(reason=reason, run=run)
    return decorator
