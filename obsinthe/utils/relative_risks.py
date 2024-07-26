import pandas as pd


class RelativeRisks:
    """The class to calculate relative risks between evidence and outcomes.

    The evidence and outcomes are expected to be encoded in one-hot encoding.

    Usage
    -----

        >>> import pandas as pd
        >>> evidence = pd.DataFrame(
        ...     index=[1, 2, 3],
        ...     columns=["flu", "smallpox"],
        ...     data=[
        ...         [1, 0],
        ...         [1, 0],
        ...         [0, 1],
        ...     ],
        ... )
        >>> outcomes = pd.DataFrame(
        ...     index=[1, 2, 3],
        ...     columns=["cough", "rash"],
        ...     data=[
        ...         [1, 0],
        ...         [1, 1],
        ...         [0, 1],
        ...     ],
        ... )
        >>> rr = RelativeRisks(evidence, outcomes)
        >>> rr.calculate
        outcomes  cough  rash
        evidence
        flu         inf   0.5
        smallpox    0.0   1.5

    Partial calculations
    ----------------------

    The class calculates the following intermediate results:

    - `sum`: The total number of instances.
    - `E_sum`: The total number of instances with evidence.
    - `non_E_sum`: The total number of instances without evidence.
    - `O_sum`: The total number of instances with outcomes.
    - `E_and_O`: The total number of instances with both evidence and outcomes.
    - `non_E_and_O`: The total number of instances with outcomes but without evidence.
    - `p_O_while_E`: The probability of outcomes given evidence.
    - `p_O_while_non_E`: The probability of outcomes given no evidence.
    - `rr`: The relative risk of outcomes given evidence.

    Masking based on specific conditions
    -------------------------------------

    The results might be influenced by the lack of evidence or outcomes
    for some instances. In this cases, it's possible to mask the results
    based on specific conditions. For example:

        >>> # Only consider evidence with more than one instance
        >>> rr.where(rr.E_sum > 1)
        outcomes  cough  rash
        evidence
        flu         inf   0.5

        >>> # Only consider cases where evidence and outcomes intersect
        >>> # in more than one instance
        >>> rr.where(rr.E_and_O > 1)
        outcomes  cough
        evidence
        flu         inf
    """

    def __init__(self, evidence, outcomes):
        # Unify instances in evidence and outcomes.
        # it's possible that for some instance in evidence there are not outcomes
        # and vice verse. We need for the indexes in the evidence and the outcomes
        # to match.
        instances = evidence.index.union(outcomes.index)
        self.E = evidence.combine_first(pd.DataFrame(index=instances)).fillna(0)
        self.E.columns.name = "evidence"

        self.O = outcomes.combine_first(pd.DataFrame(index=instances)).fillna(0)
        self.O.columns.name = "outcomes"

    def calculate(self):
        self.sum = len(self.E)
        self.E_sum = self.E.sum(axis=0)
        self.non_E_sum = self.sum - self.E_sum
        self.O_sum = self.O.sum(axis=0)
        self.E_and_O = self.E.T.dot(self.O)
        self.non_E_and_O = self.O_sum - self.E_and_O
        self.p_O_while_E = self.E_and_O.div(self.E_sum, axis=0).fillna(0)
        self.p_O_while_non_E = self.non_E_and_O.div(self.non_E_sum, axis=0).fillna(0)
        self.rr = self.p_O_while_E.div(self.p_O_while_non_E, axis=0).fillna(0)

    def where(self, mask, rr=None):
        if rr is None:
            rr = self.rr
        if isinstance(mask, pd.DataFrame):
            ret = (
                rr.where(mask)
                .dropna(how="all", axis=0)
                .dropna(how="all", axis=1)
                .fillna(0)
            )
        else:
            if mask.index.name == "evidence":
                ret = rr.loc[mask]
            elif mask.index.name == "outcomes":
                columns = mask[mask].index
                ret = rr[columns]
            else:
                raise (
                    "Couldn't determine type of the mask. "
                    "The index name should be 'evidence' or 'outcomes'"
                )
        return ret
