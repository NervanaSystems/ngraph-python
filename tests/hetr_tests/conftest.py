import pytest

def pytest_addoption(parser):
    parser.addoption('--hetr_device', action='append', default=[],
                     help='Set hetr device (cpu, gpu, etc.)')


def pytest_generate_tests(metafunc):
    if 'hetr_device' in metafunc.fixturenames:
        metafunc.parametrize("hetr_device",
                             metafunc.config.getoption('hetr_device'))
