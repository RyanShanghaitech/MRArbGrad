from setuptools import setup, Extension
import numpy

modExt = Extension\
(
    "mrautograd.ext", 
    sources = 
    [
        './mrautograd_src/ext/core/v3.cpp',
        './mrautograd_src/ext/core/GradGen.cpp',
        './mrautograd_src/ext/mtg/mtg_functions.c',
        './mrautograd_src/ext/mtg/spline.c',
        './mrautograd_src/ext/main.cpp',
    ],
    include_dirs = ["./mrautograd_src/ext/", numpy.get_include()],
    language = 'c++',
    # extra_compile_args=["-std=c++11", "-g"],
    # extra_compile_args=["-std=c++11"],
)

setup\
(
    name = 'mrautograd',
    ext_modules = [modExt],
    packages = ["mrautograd", "mrautograd.ext", "mrautograd.ext.core", "mrautograd.ext.traj", "mrautograd.ext.mtg"],
    package_dir = {"mrautograd":"./mrautograd_src/", "mrautograd.ext":"./mrautograd_src/ext/", "mrautograd.ext.core":"./mrautograd_src/ext/core/", "mrautograd.ext.traj":"./mrautograd_src/ext/traj/", "mrautograd.ext.mtg":"./mrautograd_src/ext/mtg/"},
    include_package_data = True
)
