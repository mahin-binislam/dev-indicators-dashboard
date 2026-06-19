"""
IHDI Explorer - Inequality-adjusted Human Development Index: calculator + maps + forecasts
==========================================================================================
A student-friendly Streamlit dashboard that:
  1. Shows every input, goalpost and formula behind the HDI and the IHDI (UNDP method).
  2. PREDICTS future HDI / IHDI by forecasting the underlying indicators.
  3. Maps Bangladesh's 8 divisions and Khulna's 10 districts, with a forecast-year slider
     so the maps show projected (not just current) values. Hover any region for info.

Forecasting:
  - Life expectancy & GNI per capita are forecast from real World Bank history
    using a lag-based linear regression (no heavy ML deps).
  - Education and inequality are projected with transparent annual-trend assumptions
    you control.
  - HDI / IHDI for each future year are then computed with the standard UNDP formula.

Run locally:   streamlit run app.py
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="IHDI Explorer", page_icon="🌍", layout="wide")

GOALPOSTS = {
    "life_exp":  {"min": 20.0, "max": 85.0,    "label": "Life expectancy at birth (years)"},
    "eys":       {"min": 0.0,  "max": 18.0,    "label": "Expected years of schooling"},
    "mys":       {"min": 0.0,  "max": 15.0,    "label": "Mean years of schooling"},
    "gni":       {"min": 100.0,"max": 75000.0, "label": "GNI per capita (2017 PPP $)"},
}

WB_INDICATORS = {
    "life_exp": "SP.DYN.LE00.IN",      # Life expectancy at birth, total (years)
    "gni":      "NY.GNP.PCAP.PP.KD",   # GNI per capita, PPP (constant 2017 international $)
}

DEFAULTS = {
    "Bangladesh": {"code": "BGD", "life_exp": 72.4, "eys": 12.4, "mys": 7.4,
                    "gni": 6500.0, "ineq_health": 21.0, "ineq_edu": 33.0, "ineq_income": 17.0},
    "India":      {"code": "IND", "life_exp": 67.7, "eys": 12.6, "mys": 6.6,
                    "gni": 6950.0, "ineq_health": 24.0, "ineq_edu": 38.0, "ineq_income": 19.0},
    "Pakistan":   {"code": "PAK", "life_exp": 66.4, "eys": 8.0,  "mys": 4.5,
                    "gni": 5040.0, "ineq_health": 28.0, "ineq_edu": 43.0, "ineq_income": 23.0},
    "Nepal":      {"code": "NPL", "life_exp": 68.4, "eys": 12.9, "mys": 5.1,
                    "gni": 3880.0, "ineq_health": 22.0, "ineq_edu": 38.0, "ineq_income": 18.0},
    "Sri Lanka":  {"code": "LKA", "life_exp": 76.4, "eys": 14.1, "mys": 10.8,
                    "gni": 12600.0,"ineq_health": 14.0, "ineq_edu": 22.0, "ineq_income": 24.0},
}

GEOJSON_ADM1 = ("https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/"
                "releaseData/gbOpen/BGD/ADM1/geoBoundaries-BGD-ADM1.geojson")
GEOJSON_ADM2 = ("https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/"
                "releaseData/gbOpen/BGD/ADM2/geoBoundaries-BGD-ADM2.geojson")
GEO_NAME_KEY = "shapeName"

DIVISION_CENTROIDS = {
    "Barishal": (22.70, 90.37), "Chattogram": (22.90, 91.80), "Dhaka": (23.90, 90.30),
    "Khulna": (22.95, 89.30), "Mymensingh": (24.80, 90.40), "Rajshahi": (24.55, 88.80),
    "Rangpur": (25.75, 89.25), "Sylhet": (24.80, 91.80),
}
KHULNA_CENTROIDS = {
    "Khulna": (22.85, 89.54), "Bagerhat": (22.66, 89.79), "Satkhira": (22.72, 89.07),
    "Jashore": (23.17, 89.21), "Jhenaidah": (23.54, 89.17), "Magura": (23.49, 89.42),
    "Narail": (23.17, 89.51), "Kushtia": (23.90, 89.12), "Chuadanga": (23.64, 88.84),
    "Meherpur": (23.76, 88.63),
}

# Illustrative region-level data (editable). NOT official figures.
# 'Growth %/yr' lets the map forecast each region forward at its own rate.
DIVISION_DATA = pd.DataFrame([
    ["Dhaka",      0.690, 73.5, 7.8, 0.9],
    ["Chattogram", 0.665, 73.0, 7.2, 0.9],
    ["Khulna",     0.672, 72.8, 7.5, 0.8],
    ["Rajshahi",   0.635, 71.8, 6.6, 1.0],
    ["Rangpur",    0.602, 70.9, 5.9, 1.1],
    ["Barishal",   0.624, 71.5, 6.8, 1.0],
    ["Sylhet",     0.611, 71.2, 5.7, 1.1],
    ["Mymensingh", 0.595, 70.6, 5.6, 1.2],
], columns=["Division", "HDI (illustrative)", "Life expectancy", "Mean yrs schooling", "Growth %/yr"])

KHULNA_DATA = pd.DataFrame([
    ["Khulna",    0.680, 73.2, 7.9, 0.8],
    ["Jashore",   0.662, 72.6, 7.3, 0.9],
    ["Kushtia",   0.641, 72.0, 6.7, 1.0],
    ["Jhenaidah", 0.633, 71.8, 6.5, 1.0],
    ["Magura",    0.624, 71.5, 6.3, 1.0],
    ["Narail",    0.629, 71.6, 6.6, 1.0],
    ["Bagerhat",  0.640, 72.1, 6.8, 0.9],
    ["Satkhira",  0.612, 71.2, 6.1, 1.1],
    ["Chuadanga", 0.620, 71.4, 6.2, 1.1],
    ["Meherpur",  0.604, 70.9, 5.9, 1.2],
], columns=["District", "HDI (illustrative)", "Life expectancy", "Mean yrs schooling", "Growth %/yr"])


# ----------------------------------------------------------------------
# World Bank fetch (cached)
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24)
def fetch_wb_series(country_code, indicator_code):
    """Return (years, values) sorted ascending, or ([], []) on failure."""
    base = "https://api.worldbank.org/v2/country/"
    url = base + country_code + "/indicator/" + indicator_code + "?format=json&per_page=500"
    try:
        import requests
        rows = requests.get(url, timeout=20).json()[1]
        obs = [(int(r["date"]), float(r["value"])) for r in rows if r["value"] is not None]
        obs.sort(key=lambda x: x[0])
        return [o[0] for o in obs], [o[1] for o in obs]
    except Exception:
        return [], []


def latest_from_series(years, values, max_year):
    pairs = [(y, v) for y, v in zip(years, values) if y <= max_year]
    if not pairs:
        return None, None
    return pairs[-1][1], pairs[-1][0]


@st.cache_data(ttl=60 * 60 * 24 * 7)
def fetch_geojson(url):
    try:
        import requests
        gj = requests.get(url, timeout=30).json()
        if isinstance(gj, dict) and gj.get("features"):
            return gj
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Forecasting (lag-based linear regression, numpy only)
# ----------------------------------------------------------------------
def _fit_lag_linear(values, n_lags):
    vals = list(map(float, values))
    if len(vals) <= n_lags + 1:
        return None
    rows, ys = [], []
    for i in range(n_lags, len(vals)):
        rows.append([1.0, float(i)] + [vals[i - k] for k in range(1, n_lags + 1)])
        ys.append(vals[i])
    coef, _, _, _ = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)
    return coef


def forecast_values(values, steps, n_lags=3):
    """Forecast `steps` points beyond the last observation."""
    vals = list(map(float, values))
    if steps <= 0 or not vals:
        return []
    coef = _fit_lag_linear(vals, n_lags)
    out = []
    if coef is None:
        x = np.arange(len(vals))
        if len(vals) >= 2:
            m, c = np.polyfit(x, vals, 1)
        else:
            m, c = 0.0, vals[-1]
        for s in range(1, steps + 1):
            out.append(float(m * (len(vals) - 1 + s) + c))
        return out
    for _ in range(steps):
        idx = len(vals)
        feat = [1.0, float(idx)] + [vals[idx - k] for k in range(1, n_lags + 1)]
        pred = float(np.dot(coef, feat))
        vals.append(pred)
        out.append(pred)
    return out


def series_with_forecast(years, values, end_year, base_value, base_year, fallback_growth):
    """Build {year: value} covering up to end_year. Uses WB history+forecast when
    available; otherwise grows base_value from base_year at fallback_growth (%/yr)."""
    if years and values:
        d = {int(y): float(v) for y, v in zip(years, values)}
        last = max(d)
        if end_year > last:
            fc = forecast_values([d[y] for y in sorted(d)], end_year - last)
            for i, val in enumerate(fc, start=1):
                d[last + i] = val
        return d, True
    # fallback: synthetic growth path
    d = {}
    for y in range(base_year, end_year + 1):
        d[y] = base_value * ((1 + fallback_growth / 100.0) ** (y - base_year))
    return d, False


# ----------------------------------------------------------------------
# Name matching (handles Bangladesh spelling variants)
# ----------------------------------------------------------------------
NAME_VARIANTS = {
    "chattogram": "chittagong", "barishal": "barisal", "jashore": "jessore",
    "bogura": "bogra", "cumilla": "comilla", "nawabganj": "chapainawabganj",
}


def norm_name(s):
    if not s:
        return ""
    base = "".join(ch for ch in str(s).lower() if ch.isalnum())
    return NAME_VARIANTS.get(base, base)


def match_names(our_names, geo_names):
    geo_lookup = {}
    for g in geo_names:
        geo_lookup[norm_name(g)] = g
    return {o: geo_lookup.get(norm_name(o)) for o in our_names}


def build_map(df, name_col, value_col, geojson, center, zoom, title, centroids):
    """Choropleth if geojson matches, else bubble fallback. Hover shows all columns."""
    hover_cols = [c for c in df.columns if c != name_col]
    use_choropleth = False
    plot_df = df.copy()

    if geojson is not None:
        geo_names = [f.get("properties", {}).get(GEO_NAME_KEY) for f in geojson["features"]]
        mapping = match_names(plot_df[name_col].tolist(), geo_names)
        plot_df["_geo"] = plot_df[name_col].map(mapping)
        if plot_df["_geo"].notna().sum() > 0:
            use_choropleth = True

    if use_choropleth:
        drawn = plot_df.dropna(subset=["_geo"])
        fig = px.choropleth_mapbox(
            drawn, geojson=geojson, locations="_geo",
            featureidkey="properties." + GEO_NAME_KEY,
            color=value_col, hover_name=name_col, hover_data=hover_cols,
            color_continuous_scale="Viridis", mapbox_style="carto-positron",
            center=center, zoom=zoom, opacity=0.72,
        )
    else:
        plot_df["lat"] = plot_df[name_col].map(lambda n: centroids.get(n, (None, None))[0])
        plot_df["lon"] = plot_df[name_col].map(lambda n: centroids.get(n, (None, None))[1])
        fig = px.scatter_mapbox(
            plot_df, lat="lat", lon="lon", size=value_col, color=value_col,
            hover_name=name_col, hover_data=hover_cols, size_max=38,
            color_continuous_scale="Viridis", mapbox_style="carto-positron",
            center=center, zoom=zoom,
        )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=46, b=0), title=title)
    return fig, use_choropleth


# ----------------------------------------------------------------------
# Core IHDI math
# ----------------------------------------------------------------------
def clamp01(x):
    return max(0.0, min(1.0, x))


def dimension_index(value, key):
    g = GOALPOSTS[key]
    return clamp01((value - g["min"]) / (g["max"] - g["min"]))


def income_index(gni):
    g = GOALPOSTS["gni"]
    return clamp01((math.log(max(gni, g["min"])) - math.log(g["min"])) /
                   (math.log(g["max"]) - math.log(g["min"])))


def compute(life_exp, eys, mys, gni, a_health, a_edu, a_income):
    i_health = dimension_index(life_exp, "life_exp")
    i_eys = dimension_index(eys, "eys")
    i_mys = dimension_index(mys, "mys")
    i_edu = (i_eys + i_mys) / 2.0
    i_income = income_index(gni)

    hdi = (i_health * i_edu * i_income) ** (1.0 / 3.0)

    adj_health = i_health * (1 - a_health)
    adj_edu = i_edu * (1 - a_edu)
    adj_income = i_income * (1 - a_income)
    ihdi = (adj_health * adj_edu * adj_income) ** (1.0 / 3.0)

    overall_loss = (1 - ihdi / hdi) * 100 if hdi > 0 else 0.0
    human_inequality = (a_health + a_edu + a_income) / 3.0 * 100

    return {
        "i_health": i_health, "i_eys": i_eys, "i_mys": i_mys,
        "i_edu": i_edu, "i_income": i_income,
        "hdi": hdi, "ihdi": ihdi,
        "adj_health": adj_health, "adj_edu": adj_edu, "adj_income": adj_income,
        "overall_loss": overall_loss, "human_inequality": human_inequality,
    }


def hdi_category(hdi):
    if hdi >= 0.800: return "Very High human development"
    if hdi >= 0.700: return "High human development"
    if hdi >= 0.550: return "Medium human development"
    return "Low human development"


# ----------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Inputs")
country = st.sidebar.selectbox("Country", list(DEFAULTS.keys()))
year = st.sidebar.slider("Base year", 2000, 2023, 2022)
horizon = st.sidebar.slider("Forecast horizon (years)", 1, 15, 8)
d = DEFAULTS[country]

# Pull full historical series once (used for both current value and forecasting)
le_years, le_vals = fetch_wb_series(d["code"], WB_INDICATORS["life_exp"])
gni_years, gni_vals = fetch_wb_series(d["code"], WB_INDICATORS["gni"])

use_live = st.sidebar.checkbox("Auto-fetch Life expectancy & GNI (World Bank)", value=True)
le_default, gni_default = d["life_exp"], d["gni"]
le_note = gni_note = "default"
if use_live:
    lv, ly = latest_from_series(le_years, le_vals, year)
    gv, gy = latest_from_series(gni_years, gni_vals, year)
    if lv is not None:
        le_default, le_note = round(lv, 2), "World Bank " + str(ly)
    if gv is not None:
        gni_default, gni_note = round(gv, 1), "World Bank " + str(gy)

st.sidebar.markdown("**Health & income**")
life_exp = st.sidebar.number_input("Life expectancy (years) [" + le_note + "]",
                                   10.0, 90.0, float(le_default), 0.1)
gni = st.sidebar.number_input("GNI per capita, 2017 PPP $ [" + gni_note + "]",
                              100.0, 90000.0, float(gni_default), 10.0)

st.sidebar.markdown("**Education** (from UNDP / UNESCO)")
eys = st.sidebar.number_input("Expected years of schooling", 0.0, 18.0, float(d["eys"]), 0.1)
mys = st.sidebar.number_input("Mean years of schooling", 0.0, 15.0, float(d["mys"]), 0.1)

st.sidebar.markdown("**Inequality coefficients** (Atkinson, %)")
st.sidebar.caption("Share of each dimension lost to inequality across the population.")
a_health = st.sidebar.slider("Inequality in life expectancy (%)", 0.0, 60.0, float(d["ineq_health"]), 0.5)
a_edu = st.sidebar.slider("Inequality in education (%)", 0.0, 60.0, float(d["ineq_edu"]), 0.5)
a_income = st.sidebar.slider("Inequality in income (%)", 0.0, 60.0, float(d["ineq_income"]), 0.5)

r = compute(life_exp, eys, mys, gni, a_health / 100, a_edu / 100, a_income / 100)

# ----------------------------------------------------------------------
# Header & headline metrics
# ----------------------------------------------------------------------
st.title("🌍 IHDI Explorer — calculate, predict, map")
st.caption("Inequality-adjusted Human Development Index: full UNDP calculation, "
           "multi-year forecasts, and interactive division/district maps. "
           "Figures are estimates for learning, not official UNDP values.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("HDI (" + str(year) + ")", round(r["hdi"], 3))
m2.metric("IHDI (" + str(year) + ")", round(r["ihdi"], 3))
m3.metric("Overall loss", str(round(r["overall_loss"], 1)) + "%")
m4.metric("Category", hdi_category(r["hdi"]).split(" human")[0])

st.success("**" + country + "** — " + hdi_category(r["hdi"]) +
           "  |  HDI " + str(round(r["hdi"], 3)) +
           "  →  IHDI " + str(round(r["ihdi"], 3)) +
           "  (" + str(round(r["overall_loss"], 1)) + "% lost to inequality)")

with st.expander("📖 What is the HDI and the IHDI? (read me)", expanded=False):
    st.markdown(
        """
**Human Development Index (HDI)** summarises average achievement in three dimensions:
*a long and healthy life* (life expectancy), *knowledge* (expected & mean years of
schooling) and *a decent standard of living* (GNI per capita). The HDI is the
**geometric mean** of the three normalised indices.

The **Inequality-adjusted HDI (IHDI)** discounts each dimension by how unequally it is
distributed. With no inequality, IHDI = HDI; the gap (the **overall loss**) shows how
much inequality drags down real human development.
        """
    )

# ----------------------------------------------------------------------
# 1. Inputs & goalposts
# ----------------------------------------------------------------------
st.header("1️⃣ Inputs and goalposts")
inputs_tbl = pd.DataFrame([
    ["Life expectancy at birth", life_exp, "years", 20, 85, le_note],
    ["Expected years of schooling", eys, "years", 0, 18, "input"],
    ["Mean years of schooling", mys, "years", 0, 15, "input"],
    ["GNI per capita", gni, "2017 PPP $", 100, 75000, gni_note],
], columns=["Indicator", "Value", "Unit", "Goalpost min", "Goalpost max", "Source"])
st.dataframe(inputs_tbl, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# 2. Dimension indices
# ----------------------------------------------------------------------
st.header("2️⃣ Dimension indices (normalised 0–1)")
st.markdown(
    "Each indicator is rescaled to 0–1 with its goalposts: "
    "`index = (value − min) / (max − min)`. Income uses logarithms."
)
c1, c2 = st.columns([1, 1])
with c1:
    st.latex(r"I_{Health}=\frac{LE-20}{85-20}=" + f"{r['i_health']:.3f}")
    st.latex(r"I_{EYS}=\frac{EYS}{18}=" + f"{r['i_eys']:.3f}")
    st.latex(r"I_{MYS}=\frac{MYS}{15}=" + f"{r['i_mys']:.3f}")
with c2:
    st.latex(r"I_{Education}=\frac{I_{EYS}+I_{MYS}}{2}=" + f"{r['i_edu']:.3f}")
    st.latex(r"I_{Income}=\frac{\ln(GNI)-\ln(100)}{\ln(75000)-\ln(100)}=" + f"{r['i_income']:.3f}")

# ----------------------------------------------------------------------
# 3. HDI
# ----------------------------------------------------------------------
st.header("3️⃣ HDI — geometric mean of the three indices")
st.latex(r"HDI=\left(I_{Health}\times I_{Education}\times I_{Income}\right)^{1/3}="
         + f"{r['hdi']:.3f}")

# ----------------------------------------------------------------------
# 4. Inequality adjustment -> IHDI
# ----------------------------------------------------------------------
st.header("4️⃣ Adjust for inequality → IHDI")
st.markdown(
    "Each dimension index is multiplied by `(1 − A)`, where **A** is the Atkinson "
    "inequality measure for that dimension."
)
adj_tbl = pd.DataFrame([
    ["Health", r["i_health"], a_health, r["adj_health"]],
    ["Education", r["i_edu"], a_edu, r["adj_edu"]],
    ["Income", r["i_income"], a_income, r["adj_income"]],
], columns=["Dimension", "Index", "Inequality A (%)", "Inequality-adjusted index"])
adj_tbl["Index"] = adj_tbl["Index"].round(3)
adj_tbl["Inequality-adjusted index"] = adj_tbl["Inequality-adjusted index"].round(3)
st.dataframe(adj_tbl, use_container_width=True, hide_index=True)
st.latex(r"IHDI=\left(I^{*}_{Health}\times I^{*}_{Education}\times I^{*}_{Income}\right)^{1/3}="
         + f"{r['ihdi']:.3f}")

# ----------------------------------------------------------------------
# 5. Charts
# ----------------------------------------------------------------------
st.header("5️⃣ Visual comparison (current year)")
cc1, cc2 = st.columns([3, 2])
with cc1:
    fig = go.Figure()
    dims = ["Health", "Education", "Income"]
    fig.add_trace(go.Bar(name="Index (HDI)", x=dims,
                         y=[r["i_health"], r["i_edu"], r["i_income"]], marker_color="#4C78A8"))
    fig.add_trace(go.Bar(name="Adjusted (IHDI)", x=dims,
                         y=[r["adj_health"], r["adj_edu"], r["adj_income"]], marker_color="#E45756"))
    fig.update_layout(barmode="group", yaxis_title="Index (0–1)", height=400,
                      legend=dict(orientation="h"),
                      title="Dimension indices: before vs after inequality adjustment")
    st.plotly_chart(fig, use_container_width=True)
with cc2:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=r["overall_loss"], number={"suffix": "%"},
        title={"text": "Overall loss to inequality"},
        gauge={"axis": {"range": [0, 50]}, "bar": {"color": "#E45756"},
               "steps": [{"range": [0, 10], "color": "#d9f0d3"},
                         {"range": [10, 25], "color": "#fdebbd"},
                         {"range": [25, 50], "color": "#f7c0bb"}]}))
    gauge.update_layout(height=400)
    st.plotly_chart(gauge, use_container_width=True)

# ----------------------------------------------------------------------
# 6. PREDICTIVE: forecast HDI / IHDI forward
# ----------------------------------------------------------------------
st.header("6️⃣ Prediction — forecast HDI & IHDI")
st.markdown(
    "Life expectancy and GNI are **forecast from real World Bank history** (lag-based "
    "linear regression). Education and inequality follow the annual-trend assumptions "
    "below. HDI/IHDI are then recomputed for every future year."
)

with st.expander("🔧 Trend assumptions", expanded=True):
    t1, t2, t3 = st.columns(3)
    eys_delta = t1.number_input("Expected schooling change (yrs/yr)", -0.5, 0.5, 0.08, 0.01)
    mys_delta = t1.number_input("Mean schooling change (yrs/yr)", -0.5, 0.5, 0.10, 0.01)
    ineq_delta = t2.number_input("Inequality change (pp/yr, each dim)", -2.0, 2.0, -0.3, 0.1)
    le_fallback = t3.number_input("Life exp. growth if no data (%/yr)", -1.0, 2.0, 0.2, 0.1)
    gni_fallback = t3.number_input("GNI growth if no data (%/yr)", -2.0, 8.0, 3.0, 0.1)

end_year = year + horizon
le_dict, le_live = series_with_forecast(le_years, le_vals, end_year, life_exp, year, le_fallback)
gni_dict, gni_live = series_with_forecast(gni_years, gni_vals, end_year, gni, year, gni_fallback)

proj_rows = []
for Y in range(year, end_year + 1):
    ahead = Y - year
    le_Y = le_dict.get(Y, life_exp)
    gni_Y = gni_dict.get(Y, gni)
    eys_Y = min(18.0, max(0.0, eys + ahead * eys_delta))
    mys_Y = min(15.0, max(0.0, mys + ahead * mys_delta))
    ah = max(0.0, a_health + ahead * ineq_delta) / 100.0
    ae = max(0.0, a_edu + ahead * ineq_delta) / 100.0
    ai = max(0.0, a_income + ahead * ineq_delta) / 100.0
    rr = compute(le_Y, eys_Y, mys_Y, gni_Y, ah, ae, ai)
    proj_rows.append([Y, round(le_Y, 2), round(gni_Y, 0), round(rr["hdi"], 3),
                      round(rr["ihdi"], 3), round(rr["overall_loss"], 1)])
proj = pd.DataFrame(proj_rows, columns=["Year", "Life expectancy", "GNI per capita",
                                        "HDI", "IHDI", "Loss %"])

p1, p2, p3 = st.columns(3)
p1.metric("HDI " + str(end_year), proj["HDI"].iloc[-1],
          delta=round(proj["HDI"].iloc[-1] - r["hdi"], 3))
p2.metric("IHDI " + str(end_year), proj["IHDI"].iloc[-1],
          delta=round(proj["IHDI"].iloc[-1] - r["ihdi"], 3))
p3.metric("Projected category " + str(end_year),
          hdi_category(proj["HDI"].iloc[-1]).split(" human")[0])

fc_fig = go.Figure()
fc_fig.add_trace(go.Scatter(x=proj["Year"], y=proj["HDI"], mode="lines+markers",
                            name="HDI (forecast)", line=dict(color="#4C78A8")))
fc_fig.add_trace(go.Scatter(x=proj["Year"], y=proj["IHDI"], mode="lines+markers",
                            name="IHDI (forecast)", line=dict(color="#E45756")))
fc_fig.update_layout(height=420, yaxis_title="Index (0–1)", xaxis_title="Year",
                     legend=dict(orientation="h"),
                     title="Projected HDI & IHDI, " + str(year) + "–" + str(end_year))
st.plotly_chart(fc_fig, use_container_width=True)

# Life expectancy: historical + forecast (shows the real prediction at work)
if le_years and le_vals:
    le_hist = go.Figure()
    le_hist.add_trace(go.Scatter(x=le_years, y=le_vals, mode="lines",
                                 name="Life expectancy (history)", line=dict(color="#54A24B")))
    fut_years = [Y for Y in le_dict if Y > max(le_years)]
    le_hist.add_trace(go.Scatter(x=fut_years, y=[le_dict[y] for y in fut_years],
                                 mode="lines+markers", name="Forecast",
                                 line=dict(color="#F58518", dash="dash")))
    le_hist.update_layout(height=360, yaxis_title="Years", xaxis_title="Year",
                          legend=dict(orientation="h"),
                          title="Life expectancy: World Bank history + forecast")
    st.plotly_chart(le_hist, use_container_width=True)
else:
    st.info("Live World Bank history wasn't reachable, so life expectancy & GNI use the "
            "fallback growth assumptions above for the forecast.")

with st.expander("📄 Forecast table"):
    st.dataframe(proj, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download forecast CSV", proj.to_csv(index=False).encode("utf-8"),
                       file_name=country + "_ihdi_forecast.csv", mime="text/csv")

# ----------------------------------------------------------------------
# 7. Maps (with forecast-year slider)
# ----------------------------------------------------------------------
st.header("7️⃣ Maps — hover a region; slide to forecast")
st.caption("Boundaries from the open geoBoundaries project. Region values are illustrative "
           "and editable. Each region is projected forward at its own 'Growth %/yr'.")

adm1 = fetch_geojson(GEOJSON_ADM1)
adm2 = fetch_geojson(GEOJSON_ADM2)
if adm1 is None and adm2 is None:
    st.info("Live boundary files couldn't be reached, so maps use labelled bubble markers "
            "instead (hover still shows full info).")


def project_region(df, name_col, metric, fyear):
    """Return df with a projected metric column for the chosen forecast year."""
    ahead = fyear - year
    out = df.copy()
    is_hdi = "HDI" in metric
    proj_col = metric + " (" + str(fyear) + ")"
    growth = out["Growth %/yr"] if "Growth %/yr" in out.columns else 1.0
    projected = out[metric] * ((1 + growth / 100.0) ** ahead)
    if is_hdi:
        projected = projected.clip(upper=1.0)
    out[proj_col] = projected.round(3 if is_hdi else 1)
    return out, proj_col


tab1, tab2 = st.tabs(["🌏 Bangladesh — 8 divisions", "📍 Khulna division — districts"])

with tab1:
    div_df = st.data_editor(DIVISION_DATA, hide_index=True, use_container_width=True,
                            num_rows="fixed", key="div_editor")
    cset1, cset2, cset3 = st.columns([2, 2, 2])
    metric_cols = [c for c in div_df.columns if c not in ("Division", "Growth %/yr")]
    color_by = cset1.selectbox("Metric", metric_cols, key="div_metric")
    forecast_on = cset2.checkbox("Forecast on map", value=True, key="div_fc")
    fyear = cset3.slider("Forecast year", year, end_year, min(year + 5, end_year), key="div_year")
    if forecast_on:
        mdf, mcol = project_region(div_df, "Division", color_by, fyear)
        ttl = "Bangladesh divisions — " + color_by + " (projected " + str(fyear) + ")"
    else:
        mdf, mcol = div_df, color_by
        ttl = "Bangladesh divisions — " + color_by + " (" + str(year) + ")"
    fig_div, is_chor = build_map(mdf, "Division", mcol, adm1,
                                 center={"lat": 23.7, "lon": 90.35}, zoom=5.3,
                                 title=ttl, centroids=DIVISION_CENTROIDS)
    st.plotly_chart(fig_div, use_container_width=True)
    st.caption("Mode: " + ("GeoJSON choropleth" if is_chor else "bubble fallback") +
               ". Hover a region for current + projected values.")

with tab2:
    st.markdown("Khulna is one of Bangladesh's 8 divisions, made up of 10 districts.")
    kh_df = st.data_editor(KHULNA_DATA, hide_index=True, use_container_width=True,
                           num_rows="fixed", key="kh_editor")
    kset1, kset2, kset3 = st.columns([2, 2, 2])
    metric_cols_k = [c for c in kh_df.columns if c not in ("District", "Growth %/yr")]
    color_by_k = kset1.selectbox("Metric", metric_cols_k, key="kh_metric")
    forecast_on_k = kset2.checkbox("Forecast on map", value=True, key="kh_fc")
    fyear_k = kset3.slider("Forecast year", year, end_year, min(year + 5, end_year), key="kh_year")
    if forecast_on_k:
        kdf, kcol = project_region(kh_df, "District", color_by_k, fyear_k)
        ttl_k = "Khulna districts — " + color_by_k + " (projected " + str(fyear_k) + ")"
    else:
        kdf, kcol = kh_df, color_by_k
        ttl_k = "Khulna districts — " + color_by_k + " (" + str(year) + ")"
    fig_kh, is_chor_k = build_map(kdf, "District", kcol, adm2,
                                  center={"lat": 23.05, "lon": 89.25}, zoom=7.2,
                                  title=ttl_k, centroids=KHULNA_CENTROIDS)
    st.plotly_chart(fig_kh, use_container_width=True)
    st.caption("Mode: " + ("GeoJSON choropleth" if is_chor_k else "bubble fallback") +
               ". Hover a district for current + projected values.")

st.divider()
st.caption(
    "Methodology: UNDP Human Development Report Technical Notes. Goalposts — life "
    "expectancy 20–85, expected schooling 0–18, mean schooling 0–15, GNI per capita "
    "100–75,000 (2017 PPP $). Forecasts use lag-based linear regression on World Bank "
    "history plus transparent trend assumptions; they are estimates, not official "
    "projections. Map boundaries © geoBoundaries (open data)."
)
