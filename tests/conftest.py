import pandas as pd
import pytest


@pytest.fixture
def assert_df():
    def assertion(df, expected_str):
        def normalize(s):
            # strip only training new line at the beginning, to preserve indentation
            # of the headers
            ret = s.lstrip("\n").rstrip()
            # deal with trailing spaces that get often cleaned in editor but kept
            # in real output
            ret = "\n".join([line.rstrip() for line in ret.splitlines()])
            return ret

        with pd.option_context(
            "display.max_colwidth",
            70,
            "display.max_rows",
            None,
            "display.max_columns",
            None,
        ):
            actual_str = str(df)

        assert normalize(actual_str) == normalize(expected_str)

    return assertion


pytest.register_assert_rewrite("tests.conftest")
