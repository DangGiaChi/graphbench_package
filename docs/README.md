# Generating Documentation

## Dependencies

Install sphinx and the PyG theme.
We're going to ignore the fact that the PyG theme requires an extremely old version of Sphinx and install a newer version instead.
The second install command will show some dependency conflicts because of that, but that shouldn't be a problem.

```
pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
pip install -U sphinx==9.1.0
```


## Generating HTML Documentation

Make sure the current working directory is `docs/`.
Then run:

1. `rm -r _api _build`
2. `sphinx-apidoc --output-dir _api .. --separate --no-toc`
3. `make html` (Linux) or `make.bat html` (Windows)

The output appears in `_build/html`
