from setuptools import setup, find_packages

setup(
    name='DeepESN',
    version='1.1.5',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scipy',
        'numpy'
    ],
    description="Deep Echo State Network Python library (DeepESNpy), 2018",
    author='Luca Pedrelli',
    author_email='lucapedrelli@gmail.com',
    license='BSD 3-Clause',
    url="https://github.com/lucapedrelli/DeepESN"
)
