from setuptools import setup, find_packages

setup(
    name='DeepESN',
    version='1.2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=[
        'scipy',
        'numpy'
    ],
    extras_require={
        'ridge': ['scikit-learn'],
    },
    description="Deep Echo State Network (DeepESN) Python Library",
    author='Luca Pedrelli',
    author_email='lucapedrelli@gmail.com',
    license='BSD 3-Clause',
    url="https://github.com/lucapedrelli/DeepESN"
)
