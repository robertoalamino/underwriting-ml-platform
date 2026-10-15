import pandas as pd

def temporal_split(df:pd.DataFrame) -> dict[str,pd.DataFrame]:
    # Split LendingClub loans by origin year.
    #
    # Train:      2007–2015
    # Validation: 2016
    # Test:       2017
    # Diagnostic: 2018
    #
    # Raises ValueError for missing or unsupported origination dates.
    #
    # Returns separate training, validation, test and diagnostic Dataframes.

    dates = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    if dates.isna().any():
        raise ValueError("Missing or invalid issue_d values.")

    years = dates.dt.year

    if not years.between(2007, 2018).all():
        raise ValueError("Origination years must be between 2007 and 2018.")

    splits = {
        "train": df.loc[years <= 2015].copy(),
        "validation": df.loc[years == 2016].copy(),
        "test": df.loc[years == 2017].copy(),
        "diagnostic": df.loc[years == 2018].copy()
    }

    return splits