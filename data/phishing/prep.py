from io import BytesIO, StringIO
import os
import urllib.request

import pandas as pd
from scipy.io import arff


def prep_data(binned=False):
    colnames = ['having_IP_Address',
                'URL_Length',
                'Shortining_Service',
                'having_At_Symbol',
                'double_slash_redirecting',
                'Prefix_Suffix',
                'having_Sub_Domain',
                'SSLfinal_State',
                'Domain_registeration_length',
                'Favicon',
                'port',
                'HTTPS_token',
                'Request_URL',
                'URL_of_Anchor',
                'Links_in_tags',
                'SFH',
                'Submitting_to_email',
                'Abnormal_URL',
                'Redirect',
                'on_mouseover',
                'RightClick',
                'popUpWidnow',
                'Iframe',
                'age_of_domain',
                'DNSRecord',
                'web_traffic',
                'Page_Rank',
                'Google_Index',
                'Links_pointing_to_page',
                'Statistical_report',
                'Result']

    local_path = os.path.join(os.path.dirname(__file__), "phishing.csv")
    if os.path.exists(local_path):
        dta = pd.read_csv(local_path,
                          names=colnames,
                          index_col=False,
                          skipinitialspace=True)
        return dta

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    with urllib.request.urlopen(url) as response:
        raw = response.read()
        # SciPy's ARFF reader expects text (str) lines; pass a StringIO
        data, _ = arff.loadarff(StringIO(raw.decode("utf-8")))
    dta = pd.DataFrame(data)

    def _decode_value(value):
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return value

    # ensure columns are named consistently and convert byte-valued fields to numbers
    if len(dta.columns) == len(colnames):
        dta.columns = colnames

    for col in dta.columns:
        # decode any bytes to str
        dta[col] = dta[col].map(_decode_value)

        # try converting to numeric; collect non-numeric values and fail loudly
        converted = pd.to_numeric(dta[col], errors="coerce")
        # entries that were not null/NaN in original but became NaN after coercion
        orig_non_null = ~pd.isna(dta[col])
        coerced_na = pd.isna(converted) & orig_non_null
        if coerced_na.any():
            bad_vals = pd.unique(dta[col][coerced_na])
            sample_vals = list(bad_vals[:10])
            raise ValueError(f"Non-numeric values in column '{col}': {sample_vals} (first 10)")

        # assign converted numeric series
        dta[col] = converted

    return dta

# dta.to_pickle("dta.pkl")
