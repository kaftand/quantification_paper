
import pandas as pd


def prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"

    colnames = ["att" + str(i + 1) for i in range(9)]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta.att9 = dta.att9.replace({"not_recom": 0, "recommend": 1, "very_recom": 1, "priority": 1, "spec_prior": 1})
    target_col = "att9"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    return dta

# dta.to_pickle("dta.pkl")
