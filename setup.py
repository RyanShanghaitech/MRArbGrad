from setuptools import setup, Extension
import numpy

src = \
[
    "./mrarbgrad_src/ext/traj/ScanPlan.cpp",
    "./mrarbgrad_src/ext/mag/Mag.cpp",
    "./mrarbgrad_src/ext/main.cpp",
    "./mrarbgrad_src/ext/utility/v3.cpp"
]

ext = Extension\
(
    "mrarbgrad.ext", 
    sources = src,
    include_dirs = ["./mrarbgrad_src/ext/", numpy.get_include()],
    language = 'c++'
)

setup\
(
    name = 'mrarbgrad',
    ext_modules = [ext],
    packages = ["mrarbgrad"],
    package_dir = {"mrarbgrad":"./mrarbgrad_src/"},
    include_package_data = False
)
