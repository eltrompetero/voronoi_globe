# Voronoi Globe
Package for tiling patches of globe with Voronoi tiles.

# Installation
```bash
$ git clone https://github.com/NirajKushwaha/voronoi_globe.git
$ git clone https://github.com/eltrompetero/workspace.git

$ cd voronoi_globe
$ python3 setup.py bdist_wheel
$ pip install dist/*

$ cd ..
$ cp voronoi_globe/scripts/create_vcells.py .
$ rm -rf voronoi_globe
```

# Use
```bash
$ python3 create_vcells.py 0
```
