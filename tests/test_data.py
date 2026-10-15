import pandas as pd
import pytest

from underwriting.data import temporal_split

def test_temporal_split():

     # Small synthetic dataset with known origination dates
    df = pd.DataFrame({
        "id": [101, 102, 103, 104, 105, 106, 107],
        "issue_d": [
            "Jun-2007",
            "Dec-2015",
            "Jan-2016",
            "Dec-2016",
            "Jan-2017",
            "Dec-2017",
            "Jan-2018",
        ],
        "Default": [0, 1, 0, 1, 0, 1, 0],
    })

    splits = temporal_split(df)

    # 1. Verify that every loan belongs to the correct partition
    assert splits["train"]["id"].tolist() == [101, 102]
    assert splits["validation"]["id"].tolist() == [103, 104]
    assert splits["test"]["id"].tolist() == [105, 106]
    assert splits["diagnostic"]["id"].tolist() == [107]

    # 2. Verify that partitions are disjoint
    id_sets = [set(subset["id"]) for subset in splits.values()]

    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            assert id_sets[i].isdisjoint(id_sets[j])

    # 3. Verify that no observations have been lost
    combined_ids = set().union(*id_sets)

    assert combined_ids == set(df["id"])

    # 4. Verify chronological ordering
    train_dates = pd.to_datetime(splits["train"]["issue_d"], format="%b-%Y")
    validation_dates = pd.to_datetime(splits["validation"]["issue_d"], format="%b-%Y")
    test_dates = pd.to_datetime(splits["test"]["issue_d"], format="%b-%Y")
    diagnostic_dates = pd.to_datetime(splits["diagnostic"]["issue_d"], format="%b-%Y")

    assert train_dates.max() < validation_dates.min()
    assert validation_dates.max() < test_dates.min()
    assert test_dates.max() < diagnostic_dates.min()

def test_temporal_split_rejects_invalid_dates():

    df = pd.DataFrame({"issue_d": ["Jan-2015", "not-a-date"]})

    with pytest.raises(ValueError, match="Missing or invalid"):
        temporal_split(df)


def test_temporal_split_rejects_unsupported_years():

    df = pd.DataFrame({"issue_d": ["Jan-2015", "Jan-2019"]})

    with pytest.raises(ValueError, match="between 2007 and 2018"):
        temporal_split(df)

