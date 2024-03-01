from IPython.display import display
import pandas as pd


def display_full(df, hide_index=False):
    """Helper to avoid hidding data in table when rendering."""
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_colwidth",
        None,
        "display.max_columns",
        None,
    ):
        if hide_index:
            df = df.style.hide(axis="index")
        display(df)
