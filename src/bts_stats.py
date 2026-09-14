import pandas as pd
from src.paths import DATA_RAW

BTS_PATH = DATA_RAW / "bts_june_2025.csv"

CARRIER_NAMES = {
    "AA": "American",
    "AS": "Alaska",
    "B6": "JetBlue",
    "DL": "Delta",
    "F9": "Frontier",
    "MQ": "Envoy Air",
    "NK": "Spirit",
    "OH": "PSA Airlines",
    "OO": "SkyWest",
    "UA": "United",
    "WN": "Southwest",
    "YX": "Republic Airways",
}
NAME_TO_CODE = {v.lower(): k for k, v in CARRIER_NAMES.items()}

_df = None


def _load() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(BTS_PATH)
    return _df


def detect_carrier(question: str) -> str | None:
    q = question.lower()
    for name, code in NAME_TO_CODE.items():
        if name in q:
            return code
    for code in CARRIER_NAMES:
        if code in question.split() or f" {code} " in f" {question} ":
            return code
    return None


def on_time_summary(airport: str | None = None, carrier: str | None = None) -> dict:
    df = _load()
    subset = df
    if airport:
        subset = subset[(subset["ORIGIN"] == airport) | (subset["DEST"] == airport)]
    if carrier:
        subset = subset[subset["OP_UNIQUE_CARRIER"] == carrier]

    n = len(subset)
    if n == 0:
        return {"n_flights": 0}

    cancelled = subset["CANCELLED"] == 1
    on_time = (
        subset["ARR_DELAY_NEW"] <= 15
    )  # BTS standard: on-time = arrived within 15 min

    return {
        "n_flights": n,
        "airport": airport,
        "carrier": carrier,
        "carrier_name": CARRIER_NAMES.get(carrier, carrier),
        "cancellation_rate_pct": round(cancelled.mean() * 100, 2),
        "on_time_rate_pct": round(on_time[~cancelled].mean() * 100, 2),
        "avg_arr_delay_min": round(subset.loc[~cancelled, "ARR_DELAY_NEW"].mean(), 2),
        "avg_dep_delay_min": round(subset.loc[~cancelled, "DEP_DELAY_NEW"].mean(), 2),
        "period": "June 2025",
    }
