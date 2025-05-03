rm -r "mrautograd.egg-info"
rm -r "dist"
pip uninstall mrautograd -y

python setup.py clean --all
python -m build
pip install './dist/mrautograd-0.0.0-cp312-cp312-linux_x86_64.whl' --force-reinstall