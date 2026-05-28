
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

    local_path = os.path.join(os.path.dirname(__file__), "data.csv")
    urls = [
        local_path,
        "https://raw.githubusercontent.com/atulkum/publishing/master/Video_Games_Sales_as_at_22_Dec_2016.csv",
        "https://raw.githubusercontent.com/kwan-matt/Video-Game-Sales-with-Ratings/master/Video_Games_Sales_as_at_22_Dec_2016.csv",
        "https://raw.githubusercontent.com/zygmuntz/Video-Game-Sales-with-Ratings/master/Video_Games_Sales_as_at_22_Dec_2016.csv",
        "https://raw.githubusercontent.com/kelvins/Video-Game-Sales-with-Ratings/master/Video_Games_Sales_as_at_22_Dec_2016.csv",
        "https://github.com/prasertcbs/basic-dataset/raw/refs/heads/master/Video_Games_Sales_as_at_22_Dec_2016.csv"
    ]
    dta = _read_csv_with_fallback(urls)

    dta = dta.dropna()
    dta = dta.drop(["Name", "Publisher", "Global_Sales", "Developer"], axis=1)
    target_col = "Critic_Score"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    bins = [0, 50, 70, 80, 100]
    labels = [0, 1, 2, 3]
    dta['Critic_Score'] = pd.cut(dta['Critic_Score'], bins=bins, labels=labels)
    dta['Critic_Score'] = dta['Critic_Score'].astype("int64")

    if binned:
        for col in ['Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Count', 'User_Count']:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
