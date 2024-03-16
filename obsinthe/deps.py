DEPS = {"vis": ["plotly"], "ml": ["sklearn"]}


def check_dependencies(group):
    try:
        import importlib

        for dep in DEPS[group]:
            importlib.import_module(dep)
    except ImportError as missing_imports:
        raise ImportError(
            f"""\
    Extra dependencies required. They can be installed doing

    $ pip install "obsinthe[{group}]"

    or

    $ poetry install --extras {group}
    """
        ) from missing_imports
