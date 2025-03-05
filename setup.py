from setuptools import setup, Extension
import numpy

modExt = Extension\
(
    "g4n.ext", 
    sources = 
    [
        './src/ext/v3.cpp',
        './src/ext/GradGen.cpp',
        './src/ext/main.cpp',
    ],
    include_dirs = ["./src/ext/", numpy.get_include()],
    language = 'c++',
    extra_compile_args=["-std=c++11"],
)

setup\
(
    name = 'g4n',
    ext_modules = [modExt],
    packages = ["g4n", "g4n.ext"],
    package_dir = {"g4n":"./src/", "g4n.ext":"./src/ext/"},
    include_package_data = True
)
