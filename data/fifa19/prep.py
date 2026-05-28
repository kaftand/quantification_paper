
import os

import pandas as pd


def prep_data(binned=False):
    def _read_csv_with_fallback(paths):
        last_exc = None
        for path in paths:
            try:
                return pd.read_csv(path, header=0, skipinitialspace=True, encoding='latin1')
            except Exception as exc:
                print("Failed to read from {}: {}".format(path, exc))
                last_exc = exc
        raise last_exc

    local_path = os.path.join(os.path.dirname(__file__), "data.csv")
    urls = [
        local_path,
        "https://raw.githubusercontent.com/ameybedse/FIFA-19-complete-player-dataset/master/data.csv",
        "https://raw.githubusercontent.com/amanthedorkknight/fifa-18-data/master/data.csv",
    ]

    pos_cols = ['CB',
                'RW',
                'LW',
                'CDM',
                'LM',
                'LF',
                'RCM',
                'CF',
                'CM',
                'LAM',
                'RS',
                'ST',
                'LB',
                'RDM',
                'RCB',
                'RAM',
                'LS',
                'RM',
                'LCM',
                'LWB',
                'RF',
                'CAM',
                'LCB',
                'RWB',
                'RB',
                'LDM']

    dta = _read_csv_with_fallback(urls)

    dta = dta.drop(
        ["Unnamed: 0", "ID", "Name", "Photo", "Flag", "Loaned From", "Club", "Club Logo", "Real Face", "Nationality",
         "Release Clause", "Value", "Jersey Number"], axis=1)
    def parse_wage(x):
        try:
            s = str(x)
            s = s.lstrip('€').rstrip('K')
            return float(s)
        except Exception:
            try:
                return float(x)
            except Exception:
                return 0.0

    def parse_height(x):
        try:
            s = str(x)
            if "'" in s:
                parts = s.split("'")
                feet = float(parts[0])
                inches = float(parts[1])
                return (feet * 12 + inches) * 2.54
            return float(s)
        except Exception:
            return 0.0

    def parse_weight(t):
        try:
            if isinstance(t, (int, float)):
                return int(t)
            s = str(t)
            num = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
            return int(float(num)) if num else 0
        except Exception:
            return 0

    def parse_year_offset(t, base=2018):
        try:
            s = str(t)
            year = int(s[-4:])
            return year - base if base is not None else year
        except Exception:
            try:
                return int(t) if isinstance(t, (int, float)) else 0
            except Exception:
                return 0

    dta['Wage'] = dta['Wage'].apply(parse_wage)
    dta = dta.dropna()
    dta["Height"] = dta['Height'].apply(parse_height)
    dta["Weight"] = dta["Weight"].apply(parse_weight)
    dta["Contract Valid Until"] = dta["Contract Valid Until"].apply(lambda t: parse_year_offset(t, base=2018))
    def parse_joined(t):
        try:
            s = str(t)
            year = int(s[-4:])
            return 2018 - year
        except Exception:
            try:
                return int(t)
            except Exception:
                return 0

    dta["Joined"] = dta["Joined"].apply(parse_joined)

    for col in pos_cols:
        # make sure column exists and values are parseable
        if col not in dta.columns:
            dta[col] = 1
        try:
            dta[col] = dta[col].apply(lambda x: int(eval(x)))
        except Exception:
            dta[col] = dta[col].astype(int)

    target_col = "Wage"
    object_cols = dta.select_dtypes(include=["object"]).columns
    object_cols = [col for col in object_cols if col != target_col]
    if object_cols:
        dta = pd.get_dummies(dta, columns=object_cols)

    bins = [0, 1, 3, 10, 1000]
    labels = [1, 2, 3, 4]
    dta['Wage'] = pd.cut(dta['Wage'], bins=bins, labels=labels)
    dta['Wage'] = dta['Wage'].astype("int64")

    if binned:
        for col in list(dta)[:72]:
            if col == "Wage":
                continue
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
