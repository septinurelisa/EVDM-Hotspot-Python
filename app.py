"""app.py - Dashboard EVDM: Hotspot Kebakaran Hutan dan Lahan Indonesia 2020-2025 (Streamlit).
Jalankan:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

from evdm import (VARS_NUM, VARS_KAT, VARS_ALL, URUTAN_CONF, WARNA_UTAMA, WARNA_BIRU, RINGKAS_SENSOR,
                  label, is_num, muat_data, deskriptif_num, frekuensi_kat, batas_iqr, kekuatan, fmt, fmt_p,
                  fit_ols, tabel_koef, diagnostik)

st.set_page_config(page_title="EVDM Hotspot Indonesia", page_icon="🔥", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stSidebar"] {background: #1E3A5F;}
[data-testid="stSidebar"] * {color: #F3F4F6 !important;}
</style>""", unsafe_allow_html=True)

FRAME = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
WARNA_CONF = {"Rendah": "#FCD34D", "Nominal": "#FB923C", "Tinggi": "#B91C1C"}


@st.cache_data
def load():
    return muat_data()


@st.cache_data
def cached_fit(y, xs, log_y):
    res, dd = fit_ols(load(), y, list(xs), log_y)
    diag = diagnostik(res)
    ci = tabel_koef(res)
    samp = pd.DataFrame({"aktual": dd["_y"], "fitted": res.fittedvalues, "resid": res.resid,
                         "std": res.get_influence().resid_studentized_internal}).sample(min(8000, len(dd)), random_state=3)
    return dict(tab=ci, diag=diag, samp=samp, n=int(res.nobs), r2=res.rsquared, r2a=res.rsquared_adj,
                rse=float(np.sqrt(res.scale)), F=res.fvalue, pF=res.f_pvalue, dfm=int(res.df_model), dfr=int(res.df_resid))


hot = load()

# ---------------------------------------------------------------- NAVIGASI
st.sidebar.title("🔥 EVDM Hotspot")
st.sidebar.caption("Hotspot kebakaran hutan dan lahan Indonesia 2020–2025")
halaman = st.sidebar.radio("Menu", ["Beranda", "Dataset", "Univariat", "Bivariat", "Multivariat", "Regresi",
                                    "Peta Hotspot", "Tentang"], label_visibility="collapsed")


def box_interp(teks):
    st.markdown("**Interpretasi**")
    st.markdown(teks)


# ================================================================ BERANDA
def beranda():
    st.title("Eksplorasi dan Visualisasi Pola Hotspot Kebakaran Hutan dan Lahan di Indonesia Tahun 2020–2025")
    st.caption("Serta analisis regresi linier berganda")
    c = st.columns(4)
    c[0].metric("Jumlah observasi", f"{len(hot):,}".replace(",", "."))
    c[1].metric("Jumlah variabel", hot.shape[1] - 2)  # tahun_k dan bulan_k hanya kolom bantu
    c[2].metric("Periode data", f"{hot['tanggal'].min():%Y} – {hot['tanggal'].max():%Y}")
    c[3].metric("Median FRP (MW)", fmt(hot["frp"].median(), 1))
    a, b = st.columns(2)
    with a:
        st.subheader("Latar belakang dan tujuan")
        st.write("Kebakaran hutan dan lahan adalah masalah lingkungan yang berulang di Indonesia. Satelit dapat "
                 "mendeteksi titik panas (hotspot) sehingga polanya bisa dieksplorasi dari data.")
        st.write("**Tujuan:** (1) mendeskripsikan karakteristik hotspot, (2) mengeksplorasi hubungan antarvariabel, "
                 "(3) membangun model regresi linier berganda untuk FRP.")
        st.write("**Sumber data:** NASA FIRMS, sensor MODIS (Terra dan Aqua), Collection 6.1, wilayah Indonesia.")
        st.info("Hotspot adalah deteksi anomali termal satelit, bukan bukti kebakaran yang sudah terverifikasi. "
                "FRP adalah daya radiasi api, bukan luas area terbakar.")
    with b:
        st.subheader("Jumlah hotspot kebakaran vegetasi per tahun")
        rs = pd.read_csv(RINGKAS_SENSOR)
        fig = px.bar(rs, x="tahun", y="n", color="sensor", barmode="group",
                     color_discrete_sequence=[WARNA_UTAMA, WARNA_BIRU, "#64748B"], labels={"n": "Jumlah hotspot", "tahun": "Tahun", "sensor": "Sensor"})
        fig.update_layout(**FRAME, legend=dict(orientation="h", y=-0.2)); st.plotly_chart(fig)
    st.subheader("Ringkasan statistik utama")
    tb = pd.concat([deskriptif_num(hot[v], v) for v in ["frp", "brightness", "bright_t31"]])
    st.dataframe(tb[["Variabel", "Mean", "Median", "SD", "Min", "Max"]].round(2), hide_index=True)


# ================================================================ DATASET
def dataset():
    st.title("Dataset")
    info = pd.DataFrame({
        "Variabel": ["latitude", "longitude", "tanggal", "tahun", "bulan", "satelit", "waktu", "kelas_confidence",
                     "confidence", "brightness", "bright_t31", "scan", "track", "frp"],
        "Tipe": ["Numerik", "Numerik", "Tanggal", "Numerik (diskrit)", "Numerik (diskrit)", "Kategorik", "Kategorik",
                 "Kategorik (ordinal)", "Numerik", "Numerik", "Numerik", "Numerik", "Numerik", "Numerik"],
        "Keterangan": ["Lintang lokasi hotspot (derajat)", "Bujur lokasi hotspot (derajat)", "Tanggal pengamatan satelit",
                       "Tahun pengamatan (turunan dari tanggal)", "Bulan pengamatan (turunan dari tanggal)",
                       "Satelit pengamat: Terra atau Aqua", "Pengamatan siang atau malam",
                       "Rendah (<30), Nominal (30-79), Tinggi (>=80)", "Tingkat kepercayaan deteksi (0-100)",
                       "Suhu kecerahan band 21/22 (K)", "Suhu kecerahan band 31 (K)", "Ukuran piksel arah sapuan (km)",
                       "Ukuran piksel arah lintasan (km)", "Fire Radiative Power (MW)"]})
    with st.expander("Informasi variabel", expanded=True):
        st.dataframe(info, hide_index=True)
    st.subheader("Tabel data interaktif")
    c = st.columns(3)
    th = c[0].multiselect("Tahun", sorted(hot["tahun"].unique()))
    wk = c[1].multiselect("Waktu", ["Siang", "Malam"])
    cf = c[2].multiselect("Kelas confidence", URUTAN_CONF)
    d = hot
    if th: d = d[d["tahun"].isin(th)]
    if wk: d = d[d["waktu"].isin(wk)]
    if cf: d = d[d["kelas_confidence"].isin(cf)]
    st.caption(f"{len(d):,} baris".replace(",", ".") + " (kolom dapat diurutkan; gunakan ikon pencarian di pojok tabel)")
    tampil = d.drop(columns=["tahun_k", "bulan_k"])
    st.dataframe(tampil.head(5000), height=380)
    if len(d) > 5000: st.caption("Tabel menampilkan 5.000 baris pertama; unduhan memuat seluruh baris yang terfilter.")
    st.download_button("Unduh data (CSV)", tampil.to_csv(index=False).encode("utf-8"), "hotspot_modis_olahan.csv", "text/csv")


# ================================================================ UNIVARIAT
def univariat():
    st.title("Analisis Univariat")
    st.caption("Tujuan: melihat karakteristik satu variabel.")
    c = st.columns([1, 1, 1])
    v = c[0].selectbox("Variabel", list(VARS_ALL.values()), format_func=label, index=0)
    numerik = is_num(v)
    jenis = c[1].selectbox("Jenis grafik", ["Histogram", "Boxplot", "Density"] if numerik else ["Diagram batang"])
    uselog = c[2].checkbox("Skala log10 (data > 0)", False) if v == "frp" else False
    d = hot[hot["frp"] > 0] if uselog else hot
    if numerik:
        if v == "frp" and not uselog and jenis != "Boxplot":
            d = d[d["frp"] <= d["frp"].quantile(.99)]
        if jenis == "Histogram":
            fig = px.histogram(d, x=v, nbins=50, color_discrete_sequence=[WARNA_UTAMA], log_x=uselog)
        elif jenis == "Density":
            fig = px.histogram(d, x=v, nbins=60, histnorm="probability density", color_discrete_sequence=[WARNA_UTAMA], log_x=uselog)
        else:
            fig = px.box(d, y=v, color_discrete_sequence=[WARNA_BIRU], log_y=uselog)
        ket = " (hanya sampai persentil ke-99)" if (v == "frp" and not uselog and jenis != "Boxplot") else ""
        fig.update_layout(title=f"Distribusi {label(v)}{ket}", **FRAME, bargap=.05)
    else:
        f = frekuensi_kat(hot[v])
        fig = px.bar(f, x="Kategori", y="Frekuensi", text=f["Persen"].map(lambda p: f"{p}%"), color_discrete_sequence=[WARNA_BIRU])
        fig.update_layout(title=f"Jumlah hotspot menurut {label(v).lower()}", **FRAME)
    st.plotly_chart(fig)
    st.markdown("**Statistik**")
    if numerik:
        st.dataframe(deskriptif_num(hot[v], v).round(2), hide_index=True)
        x = hot[v]; sk = ((x - x.mean()) / x.std(ddof=0)).pow(3).mean(); b0, b1 = batas_iqr(x)
        no = int(((x < b0) | (x > b1)).sum())
        bentuk = "cukup simetris" if abs(sk) < .5 else ("menjulur ke kanan (right-skewed)" if sk > 0 else "menjulur ke kiri (left-skewed)")
        tambah = " (rata-rata lebih besar dari median, tanda ekor kanan panjang)" if x.mean() > x.median() * 1.1 else ""
        box_interp(f"Distribusi **{label(v)}** {bentuk} (skewness = {fmt(sk, 2)}). Rata-rata {fmt(x.mean(), 2)}, median {fmt(x.median(), 2)}{tambah}.\n\n"
                   f"Dengan aturan 1,5 × IQR terdapat **{no:,}** pengamatan di luar batas ({fmt(100 * no / len(x), 1)}%), yaitu kemungkinan pencilan.".replace(",", "."))
    else:
        f = frekuensi_kat(hot[v]); st.dataframe(f, hide_index=True); t = f.loc[f["Frekuensi"].idxmax()]
        box_interp(f"Kategori terbanyak pada **{label(v)}** adalah **{t['Kategori']}** dengan {int(t['Frekuensi']):,} hotspot ({fmt(t['Persen'], 1)}%).".replace(",", "."))


# ================================================================ BIVARIAT
def bivariat():
    st.title("Analisis Bivariat")
    st.caption("Tujuan: melihat hubungan dua variabel. Jenis grafik menyesuaikan tipe variabel.")
    c = st.columns(3)
    x = c[0].selectbox("Variabel pertama", list(VARS_ALL.values()), format_func=label, index=1)
    y = c[1].selectbox("Variabel kedua", list(VARS_ALL.values()), format_func=label, index=0)
    uselog = c[2].checkbox("Skala log pada FRP (jika ada)", True)
    if x == y:
        st.warning("Pilih dua variabel yang berbeda."); return
    d = hot[hot["frp"] > 0] if (uselog and "frp" in (x, y)) else hot
    nx, ny = is_num(x), is_num(y)
    if nx and ny:
        s = d.sample(min(15000, len(d)), random_state=1)
        fig = px.scatter(s, x=x, y=y, opacity=.25, color_discrete_sequence=[WARNA_BIRU], log_x=uselog and x == "frp", log_y=uselog and y == "frp")
        xs = np.linspace(s[x].min(), s[x].max(), 50)
        yy = np.log10(s[y]) if (uselog and y == "frp") else s[y]
        xx = np.log10(s[x]) if (uselog and x == "frp") else s[x]
        b1, b0 = np.polyfit(xx, yy, 1)
        xl = np.log10(xs) if (uselog and x == "frp") else xs
        yl = b0 + b1 * xl
        fig.add_trace(go.Scatter(x=xs, y=10 ** yl if (uselog and y == "frp") else yl, mode="lines", name="Garis regresi sederhana", line=dict(color=WARNA_UTAMA, width=3)))
        fig.update_layout(title=f"{label(y)} vs {label(x)} (sampel acak 15.000 titik)", **FRAME)
        rp, rs_ = d[x].corr(d[y]), d[x].corr(d[y], method="spearman")
        tab = pd.DataFrame({"Ukuran": ["Korelasi Pearson", "Korelasi Spearman", "n"], "Nilai": [fmt(rp, 4), fmt(rs_, 4), f"{len(d):,}".replace(",", ".")]})
        arah = "positif (naik bersama)" if rs_ > 0 else "negatif (berlawanan arah)"
        teks = (f"Hubungan **{arah}** dengan kekuatan **{kekuatan(rs_)}** (Spearman = {fmt(rs_, 2)}; Pearson = {fmt(rp, 2)}).\n\n"
                "Jika Pearson dan Spearman berbeda jauh, hubungan kemungkinan tidak linier atau dipengaruhi pencilan.")
    elif nx != ny:
        kat, num = (y, x) if nx else (x, y)
        urut = URUTAN_CONF if kat == "kelas_confidence" else None
        fig = px.box(d, x=kat, y=num, color=kat, log_y=uselog and num == "frp", category_orders={kat: urut} if urut else None,
                     color_discrete_sequence=px.colors.sequential.Oranges[2:])
        fig.update_layout(title=f"{label(num)} menurut {label(kat).lower()}", showlegend=False, **FRAME)
        med = d.groupby(kat, observed=True)[num].median().reset_index(name="Median").rename(columns={kat: "Kelompok"})
        grp = [g[num].values for _, g in d.groupby(kat, observed=True)]
        if len(grp) == 2:
            p = stats.mannwhitneyu(*grp, method="asymptotic").pvalue; uji = "Wilcoxon"
        else:
            p = stats.kruskal(*grp).pvalue; uji = "Kruskal-Wallis"
        med["Kelompok"] = med["Kelompok"].astype(str); tab = med.copy()
        tab.loc[len(tab)] = [f"Uji {uji} (p-value)", p]; tab["Median"] = tab["Median"].map(lambda v_: fmt(v_, 4))
        mx, mn = med.loc[med["Median"].idxmax()], med.loc[med["Median"].idxmin()]
        teks = (f"Median **{label(num)}** tertinggi pada **{mx['Kelompok']}** ({fmt(mx['Median'], 2)}) dan terendah pada **{mn['Kelompok']}** ({fmt(mn['Median'], 2)}). "
                f"Uji {uji}: p-value {fmt_p(p)}, {'ada perbedaan antarkelompok' if p < .05 else 'tidak ada perbedaan yang nyata'}.\n\n"
                "Catatan: dengan data besar perbedaan kecil pun bisa signifikan; lihat selisih mediannya.")
    else:
        ct = pd.crosstab(d[x], d[y]); prop = ct.div(ct.sum(1), axis=0).reset_index().melt(id_vars=x, value_name="Proporsi")
        fig = px.bar(prop, x=x, y="Proporsi", color=y, color_discrete_sequence=px.colors.sequential.Oranges[2:])
        fig.update_layout(title=f"Proporsi {label(y).lower()} menurut {label(x).lower()}", **FRAME)
        chi2, p, _, _ = stats.chi2_contingency(ct); v_ = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        tab = pd.DataFrame({"Ukuran": ["Chi-kuadrat", "p-value", "Cramer's V"], "Nilai": [fmt(chi2, 2), fmt_p(p), fmt(v_, 3)]})
        teks = f"Uji Chi-kuadrat p-value {fmt_p(p)} ({'ada asosiasi' if p < .05 else 'tidak ada asosiasi nyata'}). Cramer's V = {fmt(v_, 3)}, asosiasi **{kekuatan(v_)}**."
    st.plotly_chart(fig)
    a, b = st.columns(2)
    with a:
        st.markdown("**Statistik hubungan**"); st.dataframe(tab, hide_index=True)
    with b:
        box_interp(teks)


# ================================================================ MULTIVARIAT
def multivariat():
    st.title("Analisis Multivariat")
    t1, t2 = st.tabs(["Heatmap korelasi", "Scatterplot multivariat"])
    with t1:
        c = st.columns([3, 1])
        vs = c[0].multiselect("Variabel numerik", list(VARS_NUM.values()), default=list(VARS_NUM.values()), format_func=label)
        met = c[1].radio("Metode korelasi", ["spearman", "pearson"], format_func=str.title)
        if len(vs) < 2:
            st.warning("Pilih minimal 2 variabel."); return
        k = hot[vs].corr(method=met)
        fig = px.imshow(k, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=[WARNA_BIRU, "white", WARNA_UTAMA], aspect="auto")
        fig.update_layout(title=f"Heatmap korelasi {met.title()}", **FRAME); st.plotly_chart(fig)
        k2 = k.where(np.triu(np.ones(k.shape), 1).astype(bool)).stack()
        kuat, lemah = k2.abs().idxmax(), k2.abs().idxmin()
        box_interp(f"Pasangan dengan hubungan terkuat: **{kuat[0]}** dan **{kuat[1]}** (r = {fmt(k2[kuat], 2)}, {kekuatan(k2[kuat])}). "
                   f"Terlemah: **{lemah[0]}** dan **{lemah[1]}** (r = {fmt(k2[lemah], 2)}).\n\n"
                   "Pasangan yang berkorelasi sangat tinggi (misalnya scan dan track) berpotensi menimbulkan multikolinieritas bila masuk model bersamaan.")
    with t2:
        c = st.columns(4); nums = list(VARS_NUM.values())
        x = c[0].selectbox("Sumbu X", nums, format_func=label, index=1, key="mx")
        y = c[1].selectbox("Sumbu Y", nums, format_func=label, index=0, key="my")
        w = c[2].selectbox("Warna", nums, format_func=label, index=3, key="mw")
        u = c[3].selectbox("Ukuran titik", nums, format_func=label, index=2, key="mu")
        c2 = st.columns([3, 1]); n = c2[0].slider("Jumlah titik (sampel acak)", 500, 8000, 3000, step=500)
        uselog = c2[1].checkbox("Skala log pada FRP", True, key="mlog")
        d = hot[hot["frp"] > 0] if (uselog and "frp" in (x, y)) else hot
        s = d.sample(min(n, len(d)), random_state=7).copy()
        s["_ukuran"] = 3 + 12 * (s[u] - s[u].min()) / (s[u].max() - s[u].min() + 1e-9)
        fig = px.scatter(s, x=x, y=y, color=w, size="_ukuran", size_max=12, opacity=.55, color_continuous_scale="Inferno_r",
                         log_x=uselog and x == "frp", log_y=uselog and y == "frp", labels={w: label(w)},
                         hover_data={"_ukuran": False, "frp": True, "brightness": True, "confidence": True})
        fig.update_layout(**FRAME, height=520); st.plotly_chart(fig)
        r = [s[v].corr(s[y], method="spearman") for v in (x, w, u)]
        box_interp(f"Pada sampel ini, korelasi Spearman terhadap **{label(y)}**: **{label(x)}** = {fmt(r[0], 2)} ({kekuatan(r[0])}), "
                   f"**{label(w)}** (warna) = {fmt(r[1], 2)} ({kekuatan(r[1])}), **{label(u)}** (ukuran) = {fmt(r[2], 2)} ({kekuatan(r[2])}). "
                   "Arahkan kursor ke titik untuk melihat nilainya.")


# ================================================================ REGRESI
def regresi():
    st.title("Analisis Regresi Linier Berganda")
    st.caption("Pilih respons dan minimal dua prediktor. Model langsung diperbarui.")
    c = st.columns([1, 3, 1])
    y = c[0].selectbox("Variabel respons (Y)", list(VARS_NUM.values()), format_func=label, index=0)
    xs = c[1].multiselect("Prediktor (X), minimal 2", [v for v in VARS_NUM.values() if v != y],
                          default=[v for v in ["brightness", "bright_t31", "scan"] if v != y], format_func=label)
    lg = c[2].checkbox("Transformasi ln pada respons", True)
    if len(xs) < 2:
        st.warning("Pilih minimal 2 prediktor (selain variabel respons)."); return
    if lg and hot[y].min() < 0:
        st.warning("Transformasi ln butuh respons tidak negatif."); return
    m = cached_fit(y, tuple(xs), lg); tab = m["tab"]; dg = m["diag"]
    yl = f"ln({y.upper()})" if lg else y.upper()
    b = tab.set_index("Parameter")["Estimasi"]; nm = b.index
    suku = " ".join(f"{b.iloc[i]:+.5f} × {nm[i]}" for i in range(1, len(b)))
    st.subheader("Persamaan model")
    st.markdown(f"**Prediksi {yl} = {b.iloc[0]:.5f} {suku}**  \n<small>n = {m['n']:,}</small>".replace(",", "."), unsafe_allow_html=True)
    st.subheader("Koefisien, uji t, dan interval konfidensi 95%")
    tt = tab.rename(columns={"p_value": "p-value", "IK95_bawah": "IK95 bawah", "IK95_atas": "IK95 atas"}).copy()
    tt["p-value"] = tt["p-value"].map(fmt_p)
    st.dataframe(tt.style.format({"Estimasi": "{:.5f}", "SE": "{:.5f}", "t": "{:.2f}", "IK95 bawah": "{:.5f}", "IK95 atas": "{:.5f}"}), hide_index=True)
    a, d_ = st.columns(2)
    with a:
        st.markdown("**Ukuran model dan uji F**")
        st.dataframe(pd.DataFrame({"Ukuran": ["Statistik F", "df", "p-value (uji F)", "R-kuadrat", "Adjusted R-kuadrat", "Residual SE"],
                                   "Nilai": [fmt(m["F"], 1), f"{m['dfm']} dan {m['dfr']}", fmt_p(m["pF"]), fmt(m["r2"], 4), fmt(m["r2a"], 4), fmt(m["rse"], 4)]}),
                     hide_index=True)
    with d_:
        st.markdown("**Diagnostik**")
        st.dataframe(pd.DataFrame({"Pemeriksaan": ["VIF terbesar", "Breusch-Pagan (p-value)", "Shapiro-Wilk, sampel 5000 (p-value)", "Durbin-Watson",
                                                   "Cook's distance maksimum", "Pengamatan dengan Cook's D > 4/n"],
                                   "Hasil": [fmt(max(dg["vif"].values()), 2), fmt_p(dg["bp_p"]), fmt_p(dg["sw_p"]), fmt(dg["dw"], 3),
                                             fmt(dg["cook_max"], 3), f"{dg['cook_n']:,}".replace(",", ".")]}), hide_index=True)
    s = m["samp"]; g = st.columns(3)
    f1 = px.scatter(s, x="fitted", y="aktual", opacity=.25, color_discrete_sequence=[WARNA_UTAMA], title="Aktual vs fitted")
    lim = [min(s["fitted"].min(), s["aktual"].min()), max(s["fitted"].max(), s["aktual"].max())]
    f1.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=WARNA_BIRU), showlegend=False)); f1.update_layout(**FRAME, height=330)
    f2 = px.scatter(s, x="fitted", y="resid", opacity=.25, color_discrete_sequence=[WARNA_BIRU], title="Residual vs fitted")
    f2.add_hline(y=0, line_color=WARNA_UTAMA); f2.update_layout(**FRAME, height=330)
    sq = np.sort(s["std"].values); tq = stats.norm.ppf((np.arange(1, len(sq) + 1) - .5) / len(sq))
    f3 = go.Figure([go.Scatter(x=tq, y=sq, mode="markers", marker=dict(size=4, opacity=.35, color=WARNA_BIRU), showlegend=False),
                    go.Scatter(x=[-4, 4], y=[-4, 4], mode="lines", line=dict(color=WARNA_UTAMA), showlegend=False)])
    f3.update_layout(title="Q-Q plot residual", xaxis_title="Kuantil teoretis", yaxis_title="Kuantil sampel", **FRAME, height=330)
    for col, f in zip(g, (f1, f2, f3)): col.plotly_chart(f)
    nol = [r.Parameter for r in tab.iloc[1:].itertuples() if r.IK95_bawah < 0 < r.IK95_atas]
    box_interp(
        f"- Uji F: p-value {fmt_p(m['pF'])}, jadi {'minimal satu prediktor berpengaruh nyata' if m['pF'] < .05 else 'belum ada bukti prediktor berpengaruh'}.\n"
        f"- R-kuadrat {fmt(m['r2'], 3)}: model menjelaskan sekitar {fmt(100 * m['r2'], 1)}% keragaman respons.\n"
        f"- Interval konfidensi 95% {'tidak ada yang memuat' if not nol else 'ada yang memuat'} nol; "
        f"{'semua koefisien signifikan pada taraf 5%.' if not nol else 'tidak signifikan: ' + ', '.join(nol)}\n"
        f"- VIF terbesar {fmt(max(dg['vif'].values()), 2)} ({'tidak ada indikasi multikolinieritas serius' if max(dg['vif'].values()) < 5 else 'perlu waspada multikolinieritas'}).\n"
        f"- Asumsi: Breusch-Pagan p-value {fmt_p(dg['bp_p'])} ({'ragam residual tidak konstan' if dg['bp_p'] < .05 else 'ragam residual konstan'}); "
        f"Shapiro-Wilk p-value {fmt_p(dg['sw_p'])} ({'residual tidak normal' if dg['sw_p'] < .05 else 'residual normal'}).\n"
        + ("- Karena respons di-ln, koefisien dibaca sebagai perubahan persentase: naik 1 satuan prediktor mengubah respons sekitar (exp(b) − 1) × 100%, prediktor lain tetap."
           if lg else "- Koefisien dibaca sebagai perubahan rata-rata respons per kenaikan 1 satuan prediktor, prediktor lain tetap."))


# ================================================================ PETA
def peta():
    st.title("Peta Hotspot")
    st.caption("Titik dibatasi sampel acak agar peta ringan.")
    c = st.columns(5)
    th = c[0].selectbox("Tahun", ["Semua"] + sorted(hot["tahun"].unique().tolist()), index=4)
    cf = c[1].selectbox("Kelas confidence", ["Semua"] + URUTAN_CONF)
    wk = c[2].selectbox("Waktu", ["Semua", "Siang", "Malam"])
    wr = c[3].radio("Warna titik", ["Kelas confidence", "FRP"])
    n = c[4].slider("Maksimum titik", 500, 5000, 2000, step=500)
    d = hot
    if th != "Semua": d = d[d["tahun"] == th]
    if cf != "Semua": d = d[d["kelas_confidence"] == cf]
    if wk != "Semua": d = d[d["waktu"] == wk]
    if d.empty:
        st.warning("Tidak ada data untuk filter ini."); return
    s = d.sample(min(n, len(d)), random_state=11).copy(); s["tanggal_s"] = s["tanggal"].dt.strftime("%Y-%m-%d")
    st.markdown(f"**Menampilkan {len(s):,} dari {len(d):,} hotspot**".replace(",", "."))
    kw = dict(lat="latitude", lon="longitude", hover_name="tanggal_s", hover_data={"frp": True, "confidence": True, "waktu": True, "latitude": False, "longitude": False},
              zoom=3.4, height=560, center=dict(lat=-2.5, lon=118))
    if wr == "Kelas confidence":
        kw.update(color="kelas_confidence", color_discrete_map=WARNA_CONF, category_orders={"kelas_confidence": URUTAN_CONF})
    else:
        s["log10_frp"] = np.log10(s["frp"].clip(lower=.1)); kw.update(color="log10_frp", color_continuous_scale="YlOrRd")
    fn = getattr(px, "scatter_map", None)
    if fn is not None:
        fig = fn(s, map_style="carto-positron", **kw)
    else:
        fig = px.scatter_mapbox(s, mapbox_style="carto-positron", **kw)
    fig.update_traces(marker=dict(size=6, opacity=.7)); fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)


# ================================================================ TENTANG
def tentang():
    st.title("Tentang Penelitian")
    st.table(pd.DataFrame({"Keterangan": ["Penulis", "Mata kuliah", "Topik", "Sumber data", "Data analisis", "Metode"],
        "Isi": ["Septi Nurelisa (NIM 241.06.4014), Program Studi Statistika, Universitas AKPRIND Indonesia",
                "Eksplorasi dan Visualisasi Data Multivariat (UTS)",
                "Pola hotspot kebakaran hutan dan lahan di Indonesia 2020-2025",
                "NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/), MODIS Collection 6.1 (Terra dan Aqua)",
                "Hotspot bertipe kebakaran vegetasi (type = 0), 117.070 observasi",
                "Statistik deskriptif, korelasi, uji Wilcoxon/Kruskal-Wallis/Chi-kuadrat, regresi linier berganda (OLS) dengan interval konfidensi 95%"]}).set_index("Keterangan"))
    st.subheader("Keterbatasan")
    st.markdown("""- Hotspot adalah deteksi termal satelit, bukan kebakaran yang terverifikasi di lapangan.
- FRP bukan luas area terbakar.
- Data tidak independen (hotspot berdekatan dalam ruang dan waktu), sehingga uji signifikansi cenderung terlalu optimis.
- Dengan observasi sebanyak ini, hampir semua uji signifikan; ukuran efek dan R-kuadrat lebih informatif.
- Asumsi regresi (normalitas residual, homoskedastisitas) tidak terpenuhi sempurna.
- Hanya sensor MODIS yang dipakai dalam analisis; VIIRS hanya dipakai untuk perbandingan jumlah.""")


{"Beranda": beranda, "Dataset": dataset, "Univariat": univariat, "Bivariat": bivariat, "Multivariat": multivariat,
 "Regresi": regresi, "Peta Hotspot": peta, "Tentang": tentang}[halaman]()
