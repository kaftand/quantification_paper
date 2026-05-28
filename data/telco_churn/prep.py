
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

    local_path = os.path.join(os.path.dirname(__file__), "churn.csv")
    urls = [
        local_path,
        "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]

    dta = _read_csv_with_fallback(urls)

    dta = dta.dropna()
    dta = dta.drop(["customerID"], axis=1)
    dta.Churn = dta.Churn.replace({"No": 0, "Yes": 1})
    target_col = "Churn"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    if binned:
        for col in list(dta)[:4]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
