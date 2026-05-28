
from io import BytesIO
import os
from zipfile import ZipFile
import urllib.request

import pandas as pd


def prep_data(binned=False):
    local_mat = os.path.join(os.path.dirname(__file__), "student-mat.csv")
    local_por = os.path.join(os.path.dirname(__file__), "student-por.csv")

    if os.path.exists(local_mat) and os.path.exists(local_por):
        dta1 = pd.read_csv(local_mat,
                           header=0,
                           skipinitialspace=True)

        dta2 = pd.read_csv(local_por,
                           header=0,
                           skipinitialspace=True)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        with urllib.request.urlopen(url) as response:
            zip_data = ZipFile(BytesIO(response.read()))
            with zip_data.open("student-mat.csv") as handle:
                dta1 = pd.read_csv(handle, header=0, sep=";", skipinitialspace=True)
            with zip_data.open("student-por.csv") as handle:
                dta2 = pd.read_csv(handle, header=0, sep=";", skipinitialspace=True)

    dta = pd.concat([dta1, dta2], ignore_index=True)

    dta.sex = dta.sex.replace({"M": 0, "F": 1})
    target_col = "sex"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    if binned:
        for col in ["age", "absences", "G1", "G2", "G3"]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
