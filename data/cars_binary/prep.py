
import pandas as pd


def prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

    colnames = ["buying",
                "maint",
                "doors",
                "persons",
                "lug_boot",
                "safety",
                "acc_class"]

    dta = pd.read_csv(url,
                      names=colnames,
                      index_col=False,
                      skipinitialspace=True)

    dta.acc_class = dta.acc_class.replace({"unacc": 0,
                                           "acc": 1,
                                           "good": 1,
                                           "vgood": 1})

    target_col = "acc_class"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    return dta

# dta.to_pickle("dta.pkl")
