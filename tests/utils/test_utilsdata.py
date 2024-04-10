import pandas as pd

from obsinthe.utils.data import get_row_digests


def test_get_row_digests(assert_df):
    df = pd.DataFrame(
        {
            "a": [1, 4, 7],
            "b": [2, 5, 8],
            "c": ["3", "6", "9"],
        }
    )
    df["digest"] = get_row_digests(df)
    assert_df(
        df,
        """
   a  b  c                                                            digest
0  1  2  3  d482163c892ce57c62ead02140894b7b1a2f935abbab4109dfc27627c068e10c
1  4  5  6  60eabe513302d31c4d13ca047a70ca742561caf629518323dc2833ad30d1f575
2  7  8  9  30400a2f81cc3176724edae309bf84c2e109b876c4c922eda29c271319957880
    """,
    )
