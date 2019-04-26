from setuptools import setup

setup(name='maxlikespy',
      version='0.1',
      description='Maximum Likelihood Models of Neural Spiking Data in Python',
      url='https://github.com/tcnlab/maxlikespy',
      author='Stephen Charczynski',
      author_email='scharcz@bu.edu',
      license='MIT',
      packages=['maxlikespy'],
      intall_requires=['autograd, numpy, scipy'],
      zip_safe=False)