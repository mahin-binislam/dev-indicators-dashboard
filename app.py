"""
Life Expectancy Dashboard - Streamlit app (deploy-ready)
========================================================
A public, student-friendly dashboard for development indicators.
Default: Life expectancy at birth, Bangladesh, from the World Bank API.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then deploy on Streamlit Community Cloud.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Development Indicators", page_icon="📊", layout="wide")

INDICATORS = {
    "Life expectancy at birth (years)": "SP.DYN.LE00.IN",
    "Under-5 mortality (per 1,000)": "SH.DYN.MORT",
    "Infant mortality (per 1,000)": "SP.DYN.IMRT.IN",
}
COUNTRIES = {"Bangladesh": "BGD", "India": "IND", "Pakistan": "PAK",
             "Nepal": "NPL", "Sri Lanka": "LKA"}


# ----------------------------------------------------------------------
# Data loading (cached so we don't refetch on every interaction)
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24)
def load_worldbank(country_code, indicator_code):
    base = "https://api.worldbank.org/v2/country/"
    url = base + country_code + "/indicator/" + indicator_code + "?format=json&per_page=500"
    import requests
    resp = requests.get(url, timeout=20)
    rows = resp.json()[1]
    data = [(int(r["date"]), float(r["value"])) for r in rows if r["value"] is not None]
    df = pd.DataFrame(data, columns=["year", "value"]).sort_values("year").reset_index(drop=True)
    return df


def forecast(df, horizon, n_lags=3):
    """Simple recursive lag-linear forecast (no heavy deps)."""
    work = df.copy()
    base_year = work["year"].min()
    # build training matrix
    rows = []
    vals = work["value"].values
    for i in range(n_lags, len(vals)):
        feat = [work["year"].iloc[i] - base_year] + [vals[i - k] for k in range(1, n_lags + 1)]
        rows.append(feat + [vals[i]])
    arr = np.array(rows)
    X, y = arr[:, :-1], arr[:, -1]
    A = np.c_[np.ones(len(X)), X]
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    history = df.copy()
    fyears, fvals = [], []
    for _ in range(horizon):
        last_year = history["year"].max()
        series = history["value"].values
        feat = [last_year + 1 - base_year] + [series[-k] for k in range(1, n_lags + 1)]
        x = np.r_[1.0, feat]
        yhat = float(x @ coef)
        ny = int(last_year + 1)
        fyears.append(ny); fvals.append(round(yhat, 2))
        history = pd.concat([history, pd.DataFrame({"year": [ny], "value": [yhat]})],
                            ignore_index=True)
    return pd.DataFrame({"year": fyears, "forecast": fvals})


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Controls")
country_name = st.sidebar.selectbox("Country", list(COUNTRIES.keys()))
indicator_name = st.sidebar.selectbox("Indicator", list(INDICATORS.keys()))
horizon = st.sidebar.slider("Forecast years", 1, 15, 7)

country_code = COUNTRIES[country_name]
indicator_code = INDICATORS[indicator_name]

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("📊 Development Indicators Dashboard")
st.caption("Data: World Bank Open Data. Forecasts are model estimates, not official figures.")

try:
    df = load_worldbank(country_code, indicator_code)
    if len(df) < 5:
        st.error("Not enough data for this selection.")
        st.stop()
except Exception as e:
    st.error("Could not load data from the World Bank API. Try again later. (" + str(e) + ")")
    st.stop()

fc = forecast(df, horizon)

col1, col2, col3 = st.columns(3)
col1.metric("Latest year", int(df["year"].iloc[-1]), )
col2.metric("Latest value", round(float(df["value"].iloc[-1]), 2))
col3.metric("Forecast " + str(int(fc["year"].iloc[-1])), round(float(fc["forecast"].iloc[-1]), 2))

st.subheader(indicator_name + " - " + country_name)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["year"], y=df["value"], mode="lines+markers", name="Historical"))
fig.add_trace(go.Scatter(x=fc["year"], y=fc["forecast"], mode="lines+markers",
                         name="Forecast", line=dict(dash="dash", color="crimson")))
fig.update_layout(xaxis_title="Year", yaxis_title=indicator_name, height=480,
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show data tables"):
    c1, c2 = st.columns(2)
    c1.write("**Historical**"); c1.dataframe(df, use_container_width=True)
    c2.write("**Forecast**"); c2.dataframe(fc, use_container_width=True)

st.info("Next step idea: add a choropleth map of Bangladesh's 8 divisions "
        "using division-level data + GeoJSON boundaries.")
