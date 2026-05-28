
import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"

    colnames = ["att" + str(i) for i in range(32)]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta = dta.drop(["att0"], axis=1)

    # dta.loc[dta['Class'] != "A", 'Class'] = "B"
    dta.att28 = dta.att28.replace({"CL0": 0,
                                   "CL1": 1,
                                   "CL2": 1,
                                   "CL3": 1,
                                   "CL4": 1,
                                   "CL5": 1,
                                   "CL6": 1})

    target_col = "att28"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    if binned:
        for col in list(dta)[:12]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
