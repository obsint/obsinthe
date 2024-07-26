import pandas as pd

from obsinthe.utils.relative_risks import RelativeRisks


def test_relative_risks(assert_df):
    evidence = pd.DataFrame(
        index=[1, 2, 3],
        columns=["flu", "smallpox"],
        data=[
            [1, 0],
            [1, 0],
            [0, 1],
        ],
    )
    outcomes = pd.DataFrame(
        index=[1, 2, 3],
        columns=["cough", "rash"],
        data=[
            [1, 0],
            [1, 1],
            [0, 1],
        ],
    )
    rr = RelativeRisks(evidence, outcomes)
    rr.calculate()

    # All results.
    assert_df(
        rr.rr,
        """
outcomes  cough  rash
evidence
flu         inf   0.5
smallpox    0.0   2.0
    """,
    )

    # Include only evidence and outcomes over certain thresholds.
    assert_df(
        rr.where(rr.E_sum > 1, rr.where(rr.O_sum > 1)),
        """
outcomes  cough  rash
evidence
flu         inf   0.5
    """,
    )

    # Condition on specific number of instances with evidence/outcome combination.
    assert_df(
        rr.where(rr.E_and_O > 1),
        """
outcomes  cough
evidence
flu         inf
    """,
    )
