
import os

import pandas as pd


def prep_data(binned=False):
    def _read_csv_with_fallback(paths):
        last_exc = None
        for path in paths:
            try:
                return pd.read_csv(path, header=0, skipinitialspace=True)
            except Exception as exc:
                last_exc = exc
        raise last_exc

    local_path = os.path.join(os.path.dirname(__file__), "StudentsPerformance.csv")
    urls = [
        local_path,
        "https://raw.githubusercontent.com/selva86/datasets/master/StudentsPerformance.csv",
        "https://raw.githubusercontent.com/ashwinprasadme/StudentsPerformance/master/StudentsPerformance.csv",
        "https://raw.githubusercontent.com/zygmuntz/datasets/master/StudentsPerformance.csv",
        "https://github.com/rashida048/Datasets/raw/refs/heads/master/StudentsPerformance.csv"
    ]

    dta = _read_csv_with_fallback(urls)

    target_col = "math score"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    bins = [-1, 66, 100]
    labels = [0, 1]
    dta['math score'] = pd.cut(dta['math score'], bins=bins, labels=labels)
    dta['math score'] = dta['math score'].astype("int64")

    if binned:
        for col in list(dta)[1:3]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
