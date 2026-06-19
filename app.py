"""
IHDI Explorer - Inequality-adjusted Human Development Index calculator + maps
=============================================================================
An informative, student-friendly Streamlit dashboard that shows EVERY input,
goalpost, and formula needed to compute the HDI and the IHDI (UNDP methodology),
PLUS interactive choropleth maps:
  - Bangladesh's 8 divisions (division-level data + GeoJSON boundaries)
  - A zoomed Khulna division with its 10 districts (hover to see info)

- Auto-fetches Life expectancy & GNI per capita (PPP) from the World Bank API.
- GeoJSON boundaries are fetched at runtime from the geoBoundaries project.
- Education values, inequality coefficients and the region-level numbers are
  editable (pre-filled with realistic / illustrative defaults).
- If boundaries can't load, the maps automatically fall back to bubble markers
  using built-in centroids, so hover-info always works.

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

# UNDP goalposts (minimum and maximum values) used to normalise each dimension
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

# Editable defaults per country (latest approximate UNDP / HDR figures).
# life_exp & gni get overwritten by the live World Bank fetch when available.
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

# geoBoundaries (open data) GeoJSON sources. ADM1 = divisions, ADM2 = districts.
GEOJSON_ADM1 = ("https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/"
                "releaseData/gbOpen/BGD/ADM1/geoBoundaries-BGD-ADM1.geojson")
GEOJSON_ADM2 = ("https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/"
                "releaseData/gbOpen/BGD/ADM2/geoBoundaries-BGD-ADM2.geojson")
GEO_NAME_KEY = "shapeName"  # geoBoundaries stores the admin name here

# Approximate centroids (lat, lon) used for the bubble-map fallback.
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

# Illustrative region-level data (editable in the app). NOT official figures.
DIVISION_DATA = pd.DataFrame([
    ["Dhaka",      0.690, 73.5, 7.8],
    ["Chattogram", 0.665, 73.0, 7.2],
    ["Khulna",     0.672, 72.8, 7.5],
    ["Rajshahi",   0.635, 71.8, 6.6],
    ["Rangpur",    0.602, 70.9, 5.9],
    ["Barishal",   0.624, 71.5, 6.8],
    ["Sylhet",     0.611, 71.2, 5.7],
    ["Mymensingh", 0.595, 70.6, 5.6],
], columns=["Division", "HDI (illustrative)", "Life expectancy", "Mean yrs schooling"])

KHULNA_DATA = pd.DataFrame([
    ["Khulna",    0.680, 73.2, 7.9],
    ["Jashore",   0.662, 72.6, 7.3],
    ["Kushtia",   0.641, 72.0, 6.7],
    ["Jhenaidah", 0.633, 71.8, 6.5],
    ["Magura",    0.624, 71.5, 6.3],
    ["Narail",    0.629, 71.6, 6.6],
    ["Bagerhat",  0.640, 72.1, 6.8],
    ["Satkhira",  0.612, 71.2, 6.1],
    ["Chuadanga", 0.620, 71.4, 6.2],
    ["Meherpur",  0.604, 70.9, 5.9],
], columns=["District", "HDI (illustrative)", "Life expectancy", "Mean yrs schooling"])


# ----------------------------------------------------------------------
# World Bank fetch (cached)
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24)
def fetch_wb_latest(country_code, indicator_code, max_year):
    """Return (value, year) for the latest non-null observation up to max_year, or (None, None)."""
    base = "https://api.worldbank.org/v2/country/"
    url = (base + country_code + "/indicator/" + indicator_code +
           "?format=json&per_page=500")
    try:
        import requests
        rows = requests.get(url, timeout=20).json()[1]
        obs = [(int(r["date"]), float(r["value"]))
               for r in rows if r["value"] is not None and int(r["date"]) <= max_year]
        if not obs:
            return None, None
        obs.sort(key=lambda x: x[0])
        year, val = obs[-1]
        return val, year
    except Exception:
        return None, None


@st.cache_data(ttl=60 * 60 * 24 * 7)
def fetch_geojson(url):
    """Fetch a GeoJSON FeatureCollection, or return None on failure."""
    try:
        import requests
        resp = requests.get(url, timeout=30)
        gj = resp.json()
        if isinstance(gj, dict) and gj.get("features"):
            return gj
        return None
    except Exception:
        return None


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
    """Map each of our names to the matching GeoJSON shapeName (or None)."""
    geo_lookup = {}
    for g in geo_names:
        geo_lookup[norm_name(g)] = g
    return {o: geo_lookup.get(norm_name(o)) for o in our_names}


def build_map(df, name_col, value_col, geojson, center, zoom, title, centroids):
    """Return a Plotly figure: choropleth if geojson matches, else bubble fallback."""
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
    """a_* are inequality coefficients as fractions (0-1)."""
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
year = st.sidebar.slider("Reference year", 2000, 2023, 2022)
d = DEFAULTS[country]

use_live = st.sidebar.checkbox("Auto-fetch Life expectancy & GNI (World Bank)", value=True)
le_default, gni_default = d["life_exp"], d["gni"]
le_note = gni_note = "default"
if use_live:
    le_val, le_yr = fetch_wb_latest(d["code"], WB_INDICATORS["life_exp"], year)
    gni_val, gni_yr = fetch_wb_latest(d["code"], WB_INDICATORS["gni"], year)
    if le_val is not None:
        le_default, le_note = round(le_val, 2), "World Bank " + str(le_yr)
    if gni_val is not None:
        gni_default, gni_note = round(gni_val, 1), "World Bank " + str(gni_yr)

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
st.title("🌍 IHDI Explorer")
st.caption("Inequality-adjusted Human Development Index — full UNDP calculation plus "
           "interactive division & district maps. Figures are estimates for learning, "
           "not official UNDP values.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("HDI", round(r["hdi"], 3), help="Human Development Index (no inequality adjustment)")
m2.metric("IHDI", round(r["ihdi"], 3), help="Inequality-adjusted HDI")
m3.metric("Overall loss", str(round(r["overall_loss"], 1)) + "%",
          help="How much HDI is lost due to inequality")
m4.metric("Category", hdi_category(r["hdi"]).split(" human")[0])

st.success("**" + country + "** — " + hdi_category(r["hdi"]) +
           "  |  HDI " + str(round(r["hdi"], 3)) +
           "  →  IHDI " + str(round(r["ihdi"], 3)) +
           "  (" + str(round(r["overall_loss"], 1)) + "% lost to inequality)")

# ----------------------------------------------------------------------
# What is the IHDI?
# ----------------------------------------------------------------------
with st.expander("📖 What is the HDI and the IHDI? (read me)", expanded=False):
    st.markdown(
        """
**Human Development Index (HDI)** summarises a country's average achievement in three basic dimensions:

- **A long and healthy life** — measured by *life expectancy at birth*.
- **Knowledge** — measured by *expected years of schooling* (for children) and *mean years of schooling* (for adults).
- **A decent standard of living** — measured by *GNI per capita* (2017 PPP $).

The **HDI** is the **geometric mean** of the three (normalised) dimension indices.

The **Inequality-adjusted HDI (IHDI)** discounts each dimension by how unequally it
is distributed across people. When there is no inequality, IHDI = HDI. The bigger the
gap, the more inequality is dragging down a country's real human development. The
**overall loss** is the percentage difference between HDI and IHDI.
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
    "Each indicator is rescaled to 0–1 using its goalposts: "
    "`index = (value − min) / (max − min)`. Income uses logarithms because extra income "
    "matters less at higher levels."
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
    "inequality measure for that dimension (the share lost to inequality). "
    "A is estimated from household survey / distribution data."
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
st.latex(r"\text{Overall loss}=\left(1-\frac{IHDI}{HDI}\right)\times100="
         + f"{r['overall_loss']:.1f}\\%")

# ----------------------------------------------------------------------
# 5. Charts
# ----------------------------------------------------------------------
st.header("5️⃣ Visual comparison")
cc1, cc2 = st.columns([3, 2])
with cc1:
    fig = go.Figure()
    dims = ["Health", "Education", "Income"]
    fig.add_trace(go.Bar(name="Index (HDI)", x=dims,
                         y=[r["i_health"], r["i_edu"], r["i_income"]],
                         marker_color="#4C78A8"))
    fig.add_trace(go.Bar(name="Adjusted (IHDI)", x=dims,
                         y=[r["adj_health"], r["adj_edu"], r["adj_income"]],
                         marker_color="#E45756"))
    fig.update_layout(barmode="group", yaxis_title="Index (0–1)",
                      height=400, legend=dict(orientation="h"),
                      title="Dimension indices: before vs after inequality adjustment")
    st.plotly_chart(fig, use_container_width=True)
with cc2:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r["overall_loss"],
        number={"suffix": "%"},
        title={"text": "Overall loss to inequality"},
        gauge={"axis": {"range": [0, 50]},
               "bar": {"color": "#E45756"},
               "steps": [{"range": [0, 10], "color": "#d9f0d3"},
                         {"range": [10, 25], "color": "#fdebbd"},
                         {"range": [25, 50], "color": "#f7c0bb"}]},
    ))
    gauge.update_layout(height=400)
    st.plotly_chart(gauge, use_container_width=True)

# ----------------------------------------------------------------------
# 6. Full results table + download
# ----------------------------------------------------------------------
st.header("6️⃣ Full results")
summary = pd.DataFrame([
    ["HDI", round(r["hdi"], 3)],
    ["IHDI", round(r["ihdi"], 3)],
    ["Overall loss (%)", round(r["overall_loss"], 1)],
    ["Coefficient of human inequality (%)", round(r["human_inequality"], 1)],
    ["Health index", round(r["i_health"], 3)],
    ["Education index", round(r["i_edu"], 3)],
    ["Income index", round(r["i_income"], 3)],
    ["HDI category", hdi_category(r["hdi"])],
], columns=["Metric", "Value"])
st.dataframe(summary, use_container_width=True, hide_index=True)
st.download_button("⬇️ Download results as CSV",
                   summary.to_csv(index=False).encode("utf-8"),
                   file_name=country + "_ihdi_" + str(year) + ".csv",
                   mime="text/csv")

# ----------------------------------------------------------------------
# 7. Maps: Bangladesh divisions + Khulna districts
# ----------------------------------------------------------------------
st.header("7️⃣ Maps — hover a region to see its data")
st.caption("Boundaries from the open geoBoundaries project. Region values below are "
           "illustrative and fully editable — replace them with real division/district figures.")

adm1 = fetch_geojson(GEOJSON_ADM1)
adm2 = fetch_geojson(GEOJSON_ADM2)
if adm1 is None and adm2 is None:
    st.info("Live boundary files couldn't be reached right now, so the maps below use "
            "labelled bubble markers instead (hover still shows full info).")

tab1, tab2 = st.tabs(["🌏 Bangladesh — 8 divisions", "📍 Khulna division — districts"])

with tab1:
    st.subheader("Choropleth of all 8 divisions")
    div_df = st.data_editor(DIVISION_DATA, hide_index=True, use_container_width=True,
                            num_rows="fixed", key="div_editor")
    metric_cols = [c for c in div_df.columns if c != "Division"]
    color_by = st.selectbox("Colour the map by", metric_cols, key="div_metric")
    fig_div, is_chor = build_map(
        div_df, "Division", color_by, adm1,
        center={"lat": 23.7, "lon": 90.35}, zoom=5.3,
        title="Bangladesh divisions — " + color_by, centroids=DIVISION_CENTROIDS,
    )
    st.plotly_chart(fig_div, use_container_width=True)
    st.caption("Mode: " + ("GeoJSON choropleth" if is_chor else "bubble-marker fallback") +
               ". Hover any region for its values.")

with tab2:
    st.subheader("Khulna division — 10 districts")
    st.markdown("Khulna is one of Bangladesh's 8 divisions, made up of 10 districts. "
                "Hover a district to see its indicators.")
    kh_df = st.data_editor(KHULNA_DATA, hide_index=True, use_container_width=True,
                           num_rows="fixed", key="kh_editor")
    metric_cols_k = [c for c in kh_df.columns if c != "District"]
    color_by_k = st.selectbox("Colour the map by", metric_cols_k, key="kh_metric")
    fig_kh, is_chor_k = build_map(
        kh_df, "District", color_by_k, adm2,
        center={"lat": 23.05, "lon": 89.25}, zoom=7.2,
        title="Khulna division districts — " + color_by_k, centroids=KHULNA_CENTROIDS,
    )
    st.plotly_chart(fig_kh, use_container_width=True)
    st.caption("Mode: " + ("GeoJSON choropleth" if is_chor_k else "bubble-marker fallback") +
               ". Hover any district for its values.")

st.divider()
st.caption(
    "Methodology: UNDP Human Development Report Technical Notes. "
    "Goalposts — life expectancy 20–85, expected schooling 0–18, mean schooling 0–15, "
    "GNI per capita 100–75,000 (2017 PPP $). Inequality coefficients come from "
    "household survey distribution data and are editable here for exploration. "
    "Map boundaries © geoBoundaries (open data)."
)
