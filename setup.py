from setuptools import setup, Extension
import numpy
import sys

_sources = \
[
    './mrautograd_src/ext/utility/v3.cpp',
    './mrautograd_src/ext/mag/GradGen.cpp',
    './mrautograd_src/ext/main.cpp',
    './mrautograd_src/ext/mtg/mtg_functions.cpp',
    './mrautograd_src/ext/mtg/spline.cpp'
]

modExt = Extension\
(
    "mrautograd.ext", 
    sources = _sources,
    libraries = [] if sys.platform=="win32" else ['jemalloc'],
    include_dirs = ["./mrautograd_src/ext/", numpy.get_include()],
    language = 'c++'
)

_packages = \
[
    "mrautograd", 
    "mrautograd.ext", 
    "mrautograd.ext.traj",
    "mrautograd.ext.mag", 
    "mrautograd.ext.mtg",
    "mrautograd.ext.utility",
]
_package_dir = \
{
    "mrautograd":"./mrautograd_src/", 
    "mrautograd.ext":"./mrautograd_src/ext/", 
    "mrautograd.ext.traj":"./mrautograd_src/ext/traj/",
    "mrautograd.ext.mag":"./mrautograd_src/ext/mag/", 
    "mrautograd.ext.mtg":"./mrautograd_src/ext/mtg/",
    "mrautograd.ext.utility":"./mrautograd_src/ext/utility/"
}

setup\
(
    name = 'mrautograd',
    install_requires = ["numpy", "matplotlib"],
    ext_modules = [modExt],
    packages = _packages,
    package_dir = _package_dir,
    include_package_data = True
)
