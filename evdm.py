"""evdm.py - fungsi bersama untuk script 02-05 dan app.py (Streamlit)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed" / "hotspot_modis.csv"
RINGKAS_SENSOR = ROOT / "data" / "processed" / "ringkasan_sensor_tahun.csv"
OUT_PLOTS = ROOT / "output" / "plots"
OUT_TABLES = ROOT / "output" / "tables"

WARNA_UTAMA = "#C2410C"   # oranye api
WARNA_BIRU = "#1E3A5F"
WARNA_ABU = "#6B7280"

VARS_NUM = {"FRP (MW)": "frp", "Brightness (K)": "brightness", "bright_t31 (K)": "bright_t31",
            "Confidence (0-100)": "confidence", "Scan (km)": "scan", "Track (km)": "track"}
VARS_KAT = {"Waktu pengamatan": "waktu", "Kelas confidence": "kelas_confidence",
            "Satelit": "satelit", "Tahun": "tahun_k", "Bulan": "bulan_k"}
VARS_ALL = {**VARS_NUM, **VARS_KAT}
URUTAN_CONF = ["Rendah", "Nominal", "Tinggi"]


def label(v):
    for k, x in VARS_ALL.items():
        if x == v:
            return k
    return v


def is_num(v):
    return v in VARS_NUM.values()


def muat_data():
    """Baca data olahan. Kolom tahun_k dan bulan_k = versi kategorik untuk grafik."""
    d = pd.read_csv(DATA_PROCESSED, parse_dates=["tanggal"])
    d["kelas_confidence"] = pd.Categorical(d["kelas_confidence"], URUTAN_CONF, ordered=True)
    d["waktu"] = pd.Categorical(d["waktu"], ["Siang", "Malam"], ordered=True)
    d["tahun_k"] = d["tahun"].astype(str)
    d["bulan_k"] = d["bulan"].astype(str).str.zfill(2)
    return d


def deskriptif_num(x, nama="x"):
    x = pd.Series(x).dropna()
    q1, med, q3 = x.quantile([.25, .5, .75])
    return pd.DataFrame([{
        "Variabel": nama, "n": len(x), "Mean": x.mean(), "Median": med, "SD": x.std(),
        "Min": x.min(), "Q1": q1, "Q3": q3, "Max": x.max(),
        "Skewness": float(((x - x.mean()) / x.std(ddof=0)).pow(3).mean())}])


def frekuensi_kat(x):
    t = pd.Series(x).value_counts(sort=False)
    return pd.DataFrame({"Kategori": t.index.astype(str), "Frekuensi": t.values,
                         "Persen": (100 * t.values / t.sum()).round(2)})


def batas_iqr(x):
    q1, q3 = np.percentile(x, [25, 75]); i = q3 - q1
    return q1 - 1.5 * i, q3 + 1.5 * i


def kekuatan(r):
    a = abs(r)
    if np.isnan(a): return "tidak dapat dihitung"
    return ("sangat lemah" if a < .2 else "lemah" if a < .4 else "sedang" if a < .6
            else "kuat" if a < .8 else "sangat kuat")


def fmt(x, d=3):
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_p(p):
    return "< 2,2e-16" if p < 2.2e-16 else f"{p:.2e}".replace(".", ",")


def fit_ols(d, y, xs, log_y=True):
    """OLS dengan statsmodels. Kembalikan (hasil, data_dipakai)."""
    import statsmodels.api as sm
    dd = d[[y] + list(xs)].dropna().copy()
    if log_y:
        dd = dd[dd[y] > 0]
        dd["_y"] = np.log(dd[y])
    else:
        dd["_y"] = dd[y]
    X = sm.add_constant(dd[list(xs)])
    return sm.OLS(dd["_y"], X).fit(), dd


def tabel_koef(res):
    ci = res.conf_int(0.05)
    return pd.DataFrame({"Parameter": res.params.index, "Estimasi": res.params.values,
                         "SE": res.bse.values, "t": res.tvalues.values, "p_value": res.pvalues.values,
                         "IK95_bawah": ci[0].values, "IK95_atas": ci[1].values})


def diagnostik(res, seed=5):
    """VIF, Breusch-Pagan (Koenker, sama dengan bptest di R), Shapiro-Wilk, DW, Cook's D."""
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    from scipy import stats
    X = res.model.exog; nm = res.model.exog_names
    vif = {nm[i]: variance_inflation_factor(X, i) for i in range(1, X.shape[1])}
    bp = het_breuschpagan(res.resid, X)
    rng = np.random.default_rng(seed)
    sw = stats.shapiro(rng.choice(res.resid.values, 5000, replace=False))
    cd = res.get_influence().cooks_distance[0]
    z = (res.resid - res.resid.mean()) / res.resid.std(ddof=0)
    return {"vif": vif, "bp_stat": bp[0], "bp_p": bp[1], "sw_p": sw.pvalue,
            "skew_resid": float((z ** 3).mean()), "dw": durbin_watson(res.resid),
            "cook_max": float(cd.max()), "cook_n": int((cd > 4 / len(cd)).sum()),
            "cook_gt1": int((cd > 1).sum())}
