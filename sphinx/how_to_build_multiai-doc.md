# How to build multiml-doc

## Install Sphinx
```bash
pip install Sphinx
pip install sphinx_rtd_theme
```

## Clone multiml package
```bash
git clone git@github.com:UTokyo-ICEPP/multiml.git
cd multiml
pip install -e .[pytorch,tensorflow24]
```

## Clone multiml-doc package
```bash
git clone git@github.com:UTokyo-ICEPP/multiml-doc.git
```

## directory structure
```bash
.
|- multiml
|- multiml-doc
|- workdir
```

## Build
```bash
mkdir workdir; cd workdir
sphinx-apidoc -F -e -o docs ../multiml/multiml
cp ../multiml/sphinx/* docs/
cd docs
make html
```

## Copy generated files
```bash
cp -r _build/html/* ../../multiml-doc/
```

## Push to git repository
```bash
cd ../../multiml-doc
git add .
git commit -m 'update'
git push
```
