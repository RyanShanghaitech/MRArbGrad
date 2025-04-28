from setuptools import setup, Extension
import numpy

modExt = Extension\
(
    "g4n.ext", 
    sources = 
    [
        './g4n_src/ext/core/v3.cpp',
        './g4n_src/ext/core/GradGen.cpp',
        './g4n_src/ext/main.cpp',
    ],
    include_dirs = ["./g4n_src/ext/", numpy.get_include()],
    language = 'c++',
    # extra_compile_args=["-std=c++11", "-g"],
    extra_compile_args=["-std=c++11", "-Ofast", "-msse2"],
)

setup\
(
    name = 'g4n',
    ext_modules = [modExt],
    packages = ["g4n", "g4n.ext", "g4n.ext.core", "g4n.ext.traj"],
    package_dir = {"g4n":"./g4n_src/", "g4n.ext":"./g4n_src/ext/", "g4n.ext.core":"./g4n_src/ext/core/", "g4n.ext.traj":"./g4n_src/ext/traj/"},
    include_package_data = True
)
