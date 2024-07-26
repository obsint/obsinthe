from hashlib import sha256

import pandas as pd


def get_row_digests(df: pd.DataFrame, exclude=[]) -> pd.Series:
    def hash_row(r):
        return sha256(
            ("".join(sorted(f"{k,v}" for (k, v) in r.to_dict().items()))).encode(
                "utf-8"
            )
        ).hexdigest()

    if exclude:
        df = df.drop(exclude, axis=1)

    return df.apply(hash_row, axis=1)


def one_hot_encode(df, index, column, prefix=None):
    ret = df.assign(value=1).pivot_table(
        index=index, columns=column, values="value", fill_value=0
    )
    if prefix:
        ret.columns = [f"{prefix}{c}" for c in ret.columns]
    return ret
