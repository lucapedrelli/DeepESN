from setuptools import setup, find_packages

setup(name='DeepESN',
      version='1.1.2',
      packages=find_packages(),
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