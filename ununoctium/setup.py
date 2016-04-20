from setuptools import setup, find_packages
setup(
    name = "Graphiti",
    version = "0.1",
    packages = find_packages(exclude=["tests"]),
    author='Nervana Systems',
    author_email='info@nervanasys.com',
    url='http://www.nervanasys.com',
    license='License :: Proprietary License',
)