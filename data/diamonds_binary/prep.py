
import os

import pandas as pd


def prep_data(binned=False):
    local_path = os.path.join(os.path.dirname(__file__), "diamonds.csv")
    if os.path.exists(local_path):
        dta = pd.read_csv(local_path,
                          header=0,
                          index_col=0,
                          skipinitialspace=True)
    else:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
        dta = pd.read_csv(url,
                          header=0,
                          skipinitialspace=True)

    dta.cut = dta.cut.replace({"Fair": 0,
                               "Good": 0,
                               "Very Good": 0,
                               "Premium": 0,
                               "Ideal": 1})

    dta = dta.rename(columns={"x": "xc", "y": "yc", "z": "zc"})

    target_col = "cut"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    if binned:
        for col in ['carat', 'depth', 'table', 'price', 'xc', 'yc', 'zc']:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

    # dta.to_pickle("dta.pkl")
