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
