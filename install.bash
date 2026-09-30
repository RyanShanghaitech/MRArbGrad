#!/usr/bin/env bash

rm -r *.egg-info build dist
pip uninstall mrarbgrad -y
set -e
python -m build
python -m pip install dist/*.whl
set +e
rm -r *.egg-info build dist

# Note: if "egg-info" remains, pip will think the package is right here, intead of in the site-package directory, which will cause problem when uninstall.
