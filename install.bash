rm -r "g4n.egg-info"
rm -r "dist"
pip uninstall g4n -y

python setup.py clean --all
python -m build
pip install './dist/g4n-0.0.0-cp312-cp312-linux_x86_64.whl' --force-reinstall