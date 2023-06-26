from numpy.distutils.core import setup, Extension

hogpy_module = Extension(
	'hogpy',
    extra_compile_args=['-std=c++11', '-march=native'],
    # undef_macros=['NDEBUG'],
    sources=['hogpy.cpp'])

setup(name='hog',
      version='0.2.1',
      description='This is a demo package',
      py_modules=['hog'],
      ext_modules=[hogpy_module])