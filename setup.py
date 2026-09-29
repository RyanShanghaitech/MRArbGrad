from setuptools import setup, Extension
import numpy

sources = \
[
    "./mrarbgrad_src/ext/traj/ScanPlan.cpp",
    "./mrarbgrad_src/ext/mag/Mag.cpp",
    "./mrarbgrad_src/ext/main.cpp",
    "./mrarbgrad_src/ext/utility/v3.cpp"
]

modExt = Extension\
(
    "mrarbgrad.ext", 
    sources = sources,
    include_dirs = ["./mrarbgrad_src/ext/", numpy.get_include()],
    language = 'c++'
)

setup\
(
    name = 'mrarbgrad',
    ext_modules = [modExt],
    packages = ["mrarbgrad"],
    package_dir = {"mrarbgrad":"./mrarbgrad_src/"},
    include_package_data = False
)
