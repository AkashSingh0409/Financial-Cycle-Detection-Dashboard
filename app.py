"""Financial Cycle Detection dashboard.

Run:  pip install streamlit plotly pandas numpy
      streamlit run app.py

Upload a BIS-style CSV (columns: BORROWERS_CTY, Quarter, Credit_Value, CPI_Value,
Credit_to_GDP_Gap; optional: Credit_to_GDP_Ratio) or use the built-in synthetic data.
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BOOM, CONTRACTION = 10.0, -10.0
REQUIRED = ["BORROWERS_CTY", "Quarter", "Credit_Value", "CPI_Value", "Credit_to_GDP_Gap"]
PLACEHOLDERS = ["..", "NA", "N/A", "-", ""]

st.set_page_config(page_title="Financial Cycle Detection", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Small synthetic BIS-like panel (3 economies, 2000Q1-2025Q4)."""
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2000Q1", "2025Q4", freq="Q")
    n, t = len(periods), np.arange(len(periods))
    params = {"United States": (150, .012, 14, 60, 0.0),
              "Japan": (170, .004, 16, 70, 1.5),
              "India": (50, .017, 13, 50, 3.0)}
    out = []
    for c, (base, g, amp, length, ph) in params.items():
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 1.5)
        gap = amp * np.sin(2 * np.pi * t / length + ph) + noise
        credit = 1000 * base / 100 * np.cumprod(1 + g + 0.0015 * gap + rng.normal(0, .004, n))
        cpi = 60 * np.cumprod(1 + g / 2 + rng.normal(0, .002, n))
        out.append(pd.DataFrame({
            "BORROWERS_CTY": c, "Quarter": [str(p) for p in periods],
            "Credit_Value": credit.round(2), "CPI_Value": cpi.round(2),
            "Credit_to_GDP_Gap": gap.round(2),
            "Credit_to_GDP_Ratio": (base + 0.05 * t + gap).round(2)}))
    return pd.concat(out, ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean (dedupe, numeric, per-country ffill/bfill), label phases."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")
    df = df.replace(PLACEHOLDERS, np.nan).drop_duplicates().copy()
    num = [c for c in ["Credit_Value", "CPI_Value", "Credit_to_GDP_Gap", "Credit_to_GDP_Ratio"]
           if c in df.columns]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    q = df["Quarter"].astype(str).str.strip().str.replace("-Q", "Q", regex=False)
    try:
        df["Date"] = pd.PeriodIndex(q, freq="Q").to_timestamp(how="end").normalize()
    except Exception:
        df["Date"] = pd.PeriodIndex(pd.to_datetime(df["Quarter"], errors="raise"),
                                    freq="Q").to_timestamp(how="end").normalize()
    df = df.sort_values(["BORROWERS_CTY", "Date"]).reset_index(drop=True)
    df[num] = df.groupby("BORROWERS_CTY")[num].transform(lambda s: s.ffill().bfill())
    df = df.dropna(subset=["Credit_to_GDP_Gap"])
    if df.empty:
        raise ValueError("No usable rows after cleaning (is Credit_to_GDP_Gap empty?).")
    df["Credit_Growth"] = df.groupby("BORROWERS_CTY")["Credit_Value"].pct_change(fill_method=None) * 100
    df["Cycle"] = pd.cut(df["Credit_to_GDP_Gap"], [-np.inf, CONTRACTION, BOOM, np.inf],
                         labels=["Contraction", "Stable", "Boom"])
    return df


# ----------------------------------------------------------------------------- sidebar
st.title("Financial Cycle Detection Dashboard")
st.caption("Credit-to-GDP gap with Basel III +/-10 pp boom / contraction thresholds")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload BIS dataset (.csv)", type="csv")

try:
    if upload is not None:
        raw = pd.read_csv(io.BytesIO(upload.getvalue()))
        source = f"Uploaded file: {upload.name}"
    else:
        raw = synthetic_data()
        source = "Synthetic fallback data"
    data = prepare(raw)
except Exception as exc:  # show a friendly error instead of a stack trace
    st.error(f"Could not load data: {exc}")
    st.info("Falling back to synthetic data. Required columns: " + ", ".join(REQUIRED))
    data = prepare(synthetic_data())
    source = "Synthetic fallback data"

with st.sidebar:
    st.success(source)
    country = st.selectbox("Country / Economy", sorted(data["BORROWERS_CTY"].unique()))
    dmin, dmax = data["Date"].min().date(), data["Date"].max().date()
    rng_sel = st.slider("Date range", dmin, dmax, (dmin, dmax))

d = data[(data["BORROWERS_CTY"] == country)
         & (data["Date"].dt.date >= rng_sel[0]) & (data["Date"].dt.date <= rng_sel[1])]
if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ----------------------------------------------------------------------------- KPIs
latest = d.iloc[-1]
counts = d["Cycle"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest gap (pp)", f"{latest.Credit_to_GDP_Gap:.1f}")
c2.metric("Current phase", str(latest.Cycle))
c3.metric("Boom quarters", int(counts.get("Boom", 0)))
c4.metric("Contraction quarters", int(counts.get("Contraction", 0)))

# ----------------------------------------------------------------------------- charts
has_ratio = "Credit_to_GDP_Ratio" in d.columns
fig = make_subplots(rows=2 if has_ratio else 1, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=(["Credit-to-GDP ratio (%)"] if has_ratio else [])
                    + ["Credit-to-GDP gap (pp)"])
row_gap = 2 if has_ratio else 1
if has_ratio:
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Credit_to_GDP_Ratio"], name="Ratio",
                             line=dict(color="#1f4e79")), row=1, col=1)

fig.add_trace(go.Scatter(x=d["Date"], y=d["Credit_to_GDP_Gap"], name="Gap",
                         line=dict(color="#222", width=2)), row=row_gap, col=1)
for label, color in [("Boom", "red"), ("Contraction", "green")]:
    sub = d[d["Cycle"] == label]
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Credit_to_GDP_Gap"], mode="markers", name=label,
                             marker=dict(color=color, size=6)), row=row_gap, col=1)
fig.add_hline(y=BOOM, line_dash="dash", line_color="red", row=row_gap, col=1,
              annotation_text="Boom alert (+10)")
fig.add_hline(y=CONTRACTION, line_dash="dash", line_color="green", row=row_gap, col=1,
              annotation_text="Contraction alert (-10)")
fig.add_hrect(y0=BOOM, y1=max(BOOM + 1, d["Credit_to_GDP_Gap"].max() + 1),
              fillcolor="red", opacity=0.08, line_width=0, row=row_gap, col=1)
fig.add_hrect(y0=min(CONTRACTION - 1, d["Credit_to_GDP_Gap"].min() - 1), y1=CONTRACTION,
              fillcolor="green", opacity=0.08, line_width=0, row=row_gap, col=1)
fig.update_layout(height=620 if has_ratio else 450, hovermode="x unified",
                  margin=dict(t=50, b=20), legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Credit growth (QoQ %) with 4-quarter average")
    g = d.dropna(subset=["Credit_Growth"]).copy()
    g["MA4"] = g["Credit_Growth"].rolling(4).mean()
    f2 = go.Figure()
    f2.add_bar(x=g["Date"], y=g["Credit_Growth"], name="Growth",
               marker_color=g["Cycle"].map({"Boom": "red", "Stable": "#9aa5b1",
                                            "Contraction": "green"}).astype(str))
    f2.add_scatter(x=g["Date"], y=g["MA4"], name="4Q MA", line=dict(color="black"))
    f2.update_layout(height=350, margin=dict(t=10))
    st.plotly_chart(f2, width="stretch")
with right:
    st.subheader("Phase distribution (all economies)")
    pc = data.groupby(["BORROWERS_CTY", "Cycle"], observed=False).size().reset_index(name="Quarters")
    f3 = go.Figure()
    for cyc, col in [("Boom", "red"), ("Stable", "#9aa5b1"), ("Contraction", "green")]:
        s = pc[pc["Cycle"] == cyc]
        f3.add_bar(x=s["BORROWERS_CTY"], y=s["Quarters"], name=cyc, marker_color=col)
    f3.update_layout(barmode="stack", height=350, margin=dict(t=10))
    st.plotly_chart(f3, width="stretch")

with st.expander("View filtered data"):
    st.dataframe(d.drop(columns=["Date"]), width="stretch")
    st.download_button("Download labelled data (CSV)", d.to_csv(index=False).encode(),
                       file_name=f"{country}_labelled.csv", mime="text/csv")
