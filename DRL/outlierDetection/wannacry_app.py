import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="WannaCry NHS Outlier Detection",
    page_icon="🦠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main app */
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #f1f5f9; }

    /* Sidebar - force dark background and light text */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] strong {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    [data-testid="stSidebar"] em {
        color: #64748b !important;
    }

    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }

    /* Source note */
    .source-note {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    /* Tabs */
    div[data-testid="stTabs"] button { color: #94a3b8; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #f1f5f9; border-bottom-color: #dc2626; }
</style>
""", unsafe_allow_html=True)


# ── Ground truth from NAO Report Appendix 2 ───────────────────────────────────
INFECTED = [
    "Barts Health NHS Trust", "Birmingham Community Healthcare NHS FT",
    "Blackpool Teaching Hospitals NHS FT", "Bradford District Care NHS FT",
    "Bridgewater Community Healthcare NHS FT", "Central Manchester University Hospitals NHS FT",
    "Colchester Hospital University NHS FT", "Cumbria Partnership NHS FT",
    "East and North Hertfordshire NHS Trust", "East Cheshire NHS Trust",
    "East Lancashire Teaching Hospitals NHS Trust", "Essex Partnership University NHS FT",
    "George Eliot Hospital NHS Trust", "Hampshire Hospitals NHS FT",
    "Hull and East Yorkshire Hospitals NHS Trust", "Humber NHS FT",
    "James Paget University Hospitals NHS FT", "Lancashire Care NHS FT",
    "Lancashire Teaching Hospital NHS Trust", "Mid Essex Hospital Services NHS Trust",
    "North Cumbria University Hospitals NHS Trust", "Northern Lincolnshire and Goole NHS FT",
    "Northumbria Healthcare NHS FT", "Nottinghamshire Healthcare NHS FT",
    "Plymouth Hospitals NHS Trust", "Royal Berkshire Hospital NHS FT",
    "Shrewsbury and Telford Hospital NHS Trust", "Solent NHS Trust",
    "Southport and Ormskirk Hospital NHS Trust", "The Dudley Group NHS FT",
    "United Lincolnshire Hospitals NHS Trust", "University Hospitals of Morecambe Bay NHS FT",
    "Wrightington Wigan and Leigh NHS FT", "York Teaching Hospitals NHS FT",
]

DISRUPTED = [
    "Airedale NHS FT", "Ashford and St Peters Hospitals NHS FT",
    "Barking Havering and Redbridge University Hospitals NHS Trust", "Barnsley Hospital NHS FT",
    "Bedford Hospital NHS Trust", "Bradford Teaching Hospitals NHS FT",
    "Brighton and Sussex University Hospitals NHS Trust", "Buckinghamshire Healthcare NHS FT",
    "Calderdale and Huddersfield NHS FT", "Central London Community Healthcare NHS Trust",
    "Chelsea and Westminster Hospital NHS FT", "Doncaster and Bassetlaw Hospitals NHS FT",
    "Dorset Healthcare NHS FT", "East Kent Hospitals University NHS FT",
    "Great Ormond Street Hospital NHS FT", "Greater Manchester Mental Health NHS FT",
    "Guys and St Thomas NHS FT", "Harrogate and District NHS FT",
    "Kettering General Hospital NHS FT", "Kingston Hospital NHS Trust",
    "Leeds and York Partnership NHS FT", "Leeds Community Healthcare NHS Trust",
    "Leeds Teaching Hospitals NHS Trust", "Leicestershire Partnership NHS Trust",
    "Lincolnshire Community Health Services NHS Trust", "Lincolnshire Partnership NHS Trust",
    "London North West Healthcare NHS Trust", "Luton and Dunstable NHS Trust",
    "Mid Yorkshire Hospitals NHS Trust", "Moorfields Eye Hospital NHS FT",
    "Norfolk and Norwich University Hospital NHS FT", "North West Ambulance Service NHS Trust",
    "Northampton General Hospital NHS Trust", "Northamptonshire Healthcare NHS FT",
    "Rotherham Doncaster and South Humber NHS FT", "Salford Royal NHS FT",
    "Sheffield Childrens NHS FT", "Sheffield Health and Social Care NHS FT",
    "Sheffield Teaching Hospitals NHS FT", "South West Yorkshire Partnership NHS FT",
    "South Western Ambulance Service NHS FT", "The Rotherham NHS FT",
    "University Hospitals of Leicester NHS Trust", "West Hertfordshire Hospitals NHS Trust",
    "West London Mental Health NHS Trust", "Yorkshire Ambulance Service NHS Trust",
]


# ── Data generation (seeded so results are reproducible) ──────────────────────
@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)

    rows = []

    def gauss():
        return rng.standard_normal()

    # Infected trusts: −4 % to −10 % drop (NAO: ~−6 % average)
    for name in INFECTED:
        baseline = int(200 + rng.random() * 800)
        effect = -(0.04 + rng.random() * 0.06)
        wannacry = baseline * (1 + effect) + gauss() * baseline * 0.02
        pct = (wannacry - baseline) / baseline * 100
        rows.append(dict(name=name, status="Infected", baseline=baseline,
                         wannacry=max(0, int(wannacry)), pct_change=round(pct, 1)))

    # Disrupted trusts: −0.5 % to −3 %
    for name in DISRUPTED:
        baseline = int(200 + rng.random() * 800)
        effect = -(0.005 + rng.random() * 0.025)
        wannacry = baseline * (1 + effect) + gauss() * baseline * 0.02
        pct = (wannacry - baseline) / baseline * 100
        rows.append(dict(name=name, status="Disrupted", baseline=baseline,
                         wannacry=max(0, int(wannacry)), pct_change=round(pct, 1)))

    # Unaffected trusts (~156)
    prefixes = ["North", "South", "East", "West", "Central", "Royal", "University",
                "City", "County", "General", "District", "Community", "Regional",
                "St Mary's", "St George's", "Queen Elizabeth", "Kings", "Pennine",
                "Peninsula", "Bay", "Valley", "Moor", "Forest", "Heath",
                "Cross", "Bridge", "Park", "Gate", "Hill", "Fields", "Green",
                "Lake", "River", "Coast", "Wessex", "Severn", "Trent", "Mersey"]
    suffixes = ["NHS Trust", "NHS Foundation Trust", "Teaching Hospitals NHS FT",
                "University Hospital NHS FT", "Hospitals NHS Trust", "Healthcare NHS Trust"]
    used = set(INFECTED + DISRUPTED)
    idx = 0
    total_unaffected = 236 - len(INFECTED) - len(DISRUPTED)
    for _ in range(total_unaffected):
        while True:
            candidate = f"{prefixes[idx % len(prefixes)]} {suffixes[(idx // len(prefixes)) % len(suffixes)]}"
            idx += 1
            if candidate not in used:
                used.add(candidate)
                break
        baseline = int(200 + rng.random() * 800)
        effect = gauss() * 0.018
        wannacry = baseline * (1 + effect)
        pct = (wannacry - baseline) / baseline * 100
        rows.append(dict(name=candidate, status="Unaffected", baseline=baseline,
                         wannacry=max(0, int(wannacry)), pct_change=round(pct, 1)))

    df = pd.DataFrame(rows)
    # Z-score of pct_change
    mu, sigma = df["pct_change"].mean(), df["pct_change"].std()
    df["z_score"] = (df["pct_change"] - mu) / sigma
    # Anomaly score: how far below mean (more negative = more anomalous)
    df["anomaly_score"] = -df["z_score"]
    return df, mu, sigma


# ── Detection logic ───────────────────────────────────────────────────────────
def apply_threshold(df, threshold):
    df = df.copy()
    df["detected"] = df["anomaly_score"] > threshold
    return df


def compute_metrics(df):
    tp = int(((df["detected"]) & (df["status"] == "Infected")).sum())
    fp = int(((df["detected"]) & (df["status"] != "Infected")).sum())
    fn = int(((~df["detected"]) & (df["status"] == "Infected")).sum())
    tn = int(((~df["detected"]) & (df["status"] != "Infected")).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, specificity=specificity)


# ── Colour mapping ────────────────────────────────────────────────────────────
def get_colour(row):
    if row["detected"] and row["status"] == "Infected":   return "#22c55e"   # TP – green
    if row["detected"] and row["status"] != "Infected":   return "#a855f7"   # FP – purple
    if not row["detected"] and row["status"] == "Infected": return "#f87171" # FN – light red
    if row["status"] == "Disrupted":                       return "#f97316"   # disrupted – orange
    return "#475569"                                                           # TN – slate


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

df_base, mu, sigma = generate_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🦠 WannaCry NHS Outlier Detection")
st.markdown("""
<div class="source-note">
📄 <b>Data source:</b> NAO Investigation: WannaCry cyber attack and the NHS (April 2018).
Ground-truth labels from <b>Appendix 2</b> (34 infected, 46 disrupted, 156 unaffected trusts).
Activity figures are <i>simulated</i> using the report's own effect sizes:
<b>−6 % total admissions</b> · <b>−9 % elective</b> · <b>−4 % emergency</b> at infected trusts.
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")
    threshold = st.slider(
        "Detection Threshold (Z-score)",
        min_value=0.5, max_value=3.5, value=1.8, step=0.1,
        help="Higher = stricter. Lower = more sensitive (catches more infected trusts but also flags more false positives)."
    )
    st.markdown(f"**Current threshold:** `z = {threshold:.1f}`")
    st.markdown("---")
    st.markdown("""
**How it works**

Each trust is scored by how far its activity dropped below the normal baseline during WannaCry week (12–19 May 2017).

The Z-score measures this deviation in standard deviations. Trusts with a score **above the threshold** are flagged as outliers.

**Colour key**
- 🟢 True Positive – correctly flagged infected trust  
- 🔴 False Negative – infected trust missed  
- 🟣 False Positive – unaffected trust wrongly flagged  
- 🟠 Disrupted (not infected)  
- ⬜ Unaffected  
""")
    st.markdown("---")
    st.markdown("*Data simulated from NAO report effect sizes. All 236 NHS Trusts represented.*")

# ── Apply threshold & compute metrics ─────────────────────────────────────────
df = apply_threshold(df_base, threshold)
m  = compute_metrics(df)
df["colour"] = df.apply(get_colour, axis=1)

# ── Top metric strip ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, colour in [
    (c1, "Flagged as Outlier", m["tp"] + m["fp"], "#dc2626"),
    (c2, "True Positives ✅",  m["tp"],            "#22c55e"),
    (c3, "False Positives ⚠️", m["fp"],            "#a855f7"),
    (c4, "Missed (FN) ❌",     m["fn"],            "#f87171"),
    (c5, "True Negatives",     m["tn"],            "#64748b"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value" style="color:{colour}">{value}</div>
      <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Scatter Plot", "📈 Model Performance", "📋 Trust List"])

# ════════════ TAB 1 – SCATTER PLOT ═══════════════════════════════════════════
with tab1:
    # Threshold y-position (convert z-score back to % change for the line)
    threshold_pct = -(threshold * sigma - mu)   # anomaly_score = -z, so flagged when -z > t → z < -t

    # Actually the threshold in pct_change space: anomaly_score > t  ↔  -z_score > t  ↔  z_score < -t
    # z_score = (pct - mu)/sigma  →  pct = mu + z*sigma
    threshold_line_y = mu + (-threshold) * sigma  # pct change at exactly z = -threshold

    fig = go.Figure()

    for status_group, symbol in [("Unaffected","circle"), ("Disrupted","diamond"), ("Infected","star")]:
        subset = df[df["status"] == status_group]
        fig.add_trace(go.Scatter(
            x=subset["baseline"],
            y=subset["pct_change"],
            mode="markers",
            marker=dict(
                color=subset["colour"],
                size=subset["status"].map({"Infected": 10, "Disrupted": 8, "Unaffected": 6})[subset.index],
                symbol=symbol,
                line=dict(width=0.5, color="#0f172a"),
                opacity=0.85
            ),
            name=status_group,
            customdata=np.stack([
                subset["name"],
                subset["status"],
                subset["wannacry"],
                subset["z_score"],
                subset["detected"].map({True: "⚠ FLAGGED", False: "Not flagged"})
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Status: %{customdata[1]}<br>"
                "Baseline admissions: %{x}<br>"
                "WannaCry week: %{customdata[2]}<br>"
                "% Change: %{y:.1f}%<br>"
                "Z-score: %{customdata[3]:.2f}<br>"
                "<b>%{customdata[4]}</b>"
                "<extra></extra>"
            )
        ))

    # Threshold line
    fig.add_hline(
        y=threshold_line_y,
        line_dash="dash", line_color="#dc2626", line_width=2,
        annotation_text=f"Detection threshold (z = −{threshold:.1f})",
        annotation_position="bottom right",
        annotation_font=dict(color="#dc2626", size=11)
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        title=dict(text="Admission Change (%) vs Baseline Volume — WannaCry Week vs Prior Baseline",
                   font=dict(size=14, color="#f1f5f9"), x=0),
        xaxis=dict(title="Baseline Daily Admissions", gridcolor="#1e3a5f",
                   zerolinecolor="#334155", tickfont=dict(color="#94a3b8")),
        yaxis=dict(title="% Change in Admissions", gridcolor="#1e3a5f",
                   zerolinecolor="#334155", tickfont=dict(color="#94a3b8")),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1,
                    font=dict(color="#e2e8f0")),
        height=520,
        hovermode="closest",
        margin=dict(l=60, r=30, t=50, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ Stars = infected trusts · Diamonds = disrupted · Circles = unaffected. Drag the threshold slider in the sidebar to update the detection line.")

# ════════════ TAB 2 – MODEL PERFORMANCE ══════════════════════════════════════
with tab2:
    col_cm, col_bars = st.columns([1, 1])

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm_data = {
            "": ["**Actual: Infected**", "**Actual: Not Infected**"],
            "Predicted: Infected": [f"✅ TP = {m['tp']}", f"🟣 FP = {m['fp']}"],
            "Predicted: Not Infected": [f"🔴 FN = {m['fn']}", f"⬜ TN = {m['tn']}"],
        }
        st.dataframe(pd.DataFrame(cm_data).set_index(""), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Precision–Recall Curve (across all thresholds)")

        thresholds = np.arange(0.5, 3.55, 0.1)
        pr_data = []
        for t in thresholds:
            d = apply_threshold(df_base, t)
            _m = compute_metrics(d)
            pr_data.append({"threshold": round(t, 1), "precision": _m["precision"], "recall": _m["recall"]})
        pr_df = pd.DataFrame(pr_data)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr_df["recall"], y=pr_df["precision"],
            mode="lines+markers",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=5, color="#dc2626"),
            customdata=pr_df[["threshold"]],
            hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<br>Threshold: %{customdata[0]}<extra></extra>"
        ))
        # Mark current threshold
        current_row = pr_df[pr_df["threshold"] == round(threshold, 1)]
        if not current_row.empty:
            fig_pr.add_trace(go.Scatter(
                x=current_row["recall"], y=current_row["precision"],
                mode="markers",
                marker=dict(size=14, color="#fbbf24", symbol="star"),
                name=f"Current (z={threshold:.1f})",
                hovertemplate="Current threshold<extra></extra>"
            ))
        fig_pr.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"), height=280,
            xaxis=dict(title="Recall", gridcolor="#1e3a5f", range=[0, 1]),
            yaxis=dict(title="Precision", gridcolor="#1e3a5f", range=[0, 1]),
            showlegend=False, margin=dict(l=50, r=20, t=20, b=50)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_bars:
        st.markdown("#### Performance Scores")
        scores = [
            ("Precision",   m["precision"],    "Of all flagged trusts, how many were truly infected?",   "#3b82f6"),
            ("Recall",      m["recall"],        "Of all infected trusts, how many did we catch?",         "#22c55e"),
            ("F1 Score",    m["f1"],            "Harmonic mean — balances precision & recall.",           "#f59e0b"),
            ("Specificity", m["specificity"],   "Of uninfected trusts, how many did we correctly clear?", "#8b5cf6"),
        ]
        for name, val, desc, colour in scores:
            bar_pct = int(val * 100)
            colour_bar = "#22c55e" if val > 0.75 else "#fbbf24" if val > 0.5 else "#ef4444"
            st.markdown(f"""
            <div style="margin-bottom:16px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-weight:600;color:#cbd5e1;font-size:13px">{name}</span>
                <span style="font-weight:800;color:{colour_bar};font-size:15px">{bar_pct}%</span>
              </div>
              <div style="background:#0f172a;border-radius:6px;height:10px;overflow:hidden">
                <div style="width:{bar_pct}%;height:100%;border-radius:6px;background:{colour_bar};transition:width 0.4s ease"></div>
              </div>
              <div style="font-size:11px;color:#475569;margin-top:4px">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### What does missing a trust mean?")
        st.info(f"""
**{m['fn']} infected trusts were missed** at this threshold.

In practice this means those hospitals would **not** receive:
- Targeted NHS Digital patch support
- Priority restoration teams
- Financial recovery allocation

The NAO report notes the total economic impact was **£5.9 million** across infected trusts.
Missing trusts delays recovery and prolongs patient impact.
        """)

# ════════════ TAB 3 – TRUST LIST ═════════════════════════════════════════════
with tab3:
    col_tp, col_fp, col_fn = st.columns(3)

    tp_trusts = df[(df["detected"]) & (df["status"] == "Infected")][["name","pct_change","z_score"]].sort_values("pct_change")
    fp_trusts = df[(df["detected"]) & (df["status"] != "Infected")][["name","status","pct_change","z_score"]].sort_values("pct_change")
    fn_trusts = df[(~df["detected"]) & (df["status"] == "Infected")][["name","pct_change","z_score"]].sort_values("pct_change")

    with col_tp:
        st.markdown(f"#### ✅ Correctly Detected — {len(tp_trusts)}")
        st.dataframe(
            tp_trusts.rename(columns={"name":"Trust","pct_change":"% Change","z_score":"Z-score"}),
            use_container_width=True, height=500, hide_index=True
        )

    with col_fp:
        st.markdown(f"#### 🟣 False Positives — {len(fp_trusts)}")
        st.dataframe(
            fp_trusts.rename(columns={"name":"Trust","status":"Status","pct_change":"% Change","z_score":"Z-score"}),
            use_container_width=True, height=500, hide_index=True
        )

    with col_fn:
        st.markdown(f"#### 🔴 Missed Infections — {len(fn_trusts)}")
        st.dataframe(
            fn_trusts.rename(columns={"name":"Trust","pct_change":"% Change","z_score":"Z-score"}),
            use_container_width=True, height=500, hide_index=True
        )

    st.markdown("---")
    st.download_button(
        "⬇️ Download Full Results as CSV",
        data=df[["name","status","baseline","wannacry","pct_change","z_score","anomaly_score","detected"]].to_csv(index=False),
        file_name="wannacry_outlier_results.csv",
        mime="text/csv"
    )
