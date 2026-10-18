
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Retail Insights | Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = [
    "#3980F6", "#27B77A", "#F5A623", "#9854E8",
    "#EC7187", "#18A6B5", "#7085A6", "#D18A35"
]


# =========================================================
# 2. PREMIUM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #F6F8FD;
    color: #142B50;
}

[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    max-width: 1800px;
    padding: 1.1rem 1.5rem 1.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101A35 0%, #17294D 100%);
    border-right: 1px solid #26385D;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1rem 0.85rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C7D4F0;
}

[data-testid="stSidebar"] hr {
    border-color: #344769;
    margin: 16px 0;
}

[data-testid="stSidebar"] .stRadio > label {
    color: #AFC1E5;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 6px;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 12px;
    margin: 0;
    transition: background 0.2s ease;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: #25385D;
    border-color: #354B76;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(100deg, #3457A8, #493C88);
    border: 1px solid #526CB5;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
    display: none;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #1C3054;
    border: 1px dashed #6079AD;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #304875;
    color: white;
    border: 1px solid #526B9A;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding-bottom: 3px;
}

.sidebar-note {
    background: linear-gradient(140deg, #20365D, #1A2B4C);
    border: 1px solid #3B5078;
    padding: 13px;
    border-radius: 12px;
    color: #DCE7FF;
    font-size: 12px;
    line-height: 1.9;
    box-shadow: 0 5px 15px rgba(0, 0, 0, .12);
}

/* Main header */
.hero {
    background:
        radial-gradient(circle at 90% 0%, #DDEAFF 0%, transparent 35%),
        linear-gradient(110deg, #FFFFFF 0%, #F0F3FF 65%, #EDF7FF 100%);
    border: 1px solid #E5EAF6;
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: clamp(22px, 2.2vw, 32px);
    color: #102A52;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.7px;
}

.hero p {
    color: #58739F;
    margin: 7px 0 0 0;
    font-size: 13px;
}

.hero-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #5B66CE;
    margin-bottom: 7px;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF5;
    border-radius: 14px;
    padding: 16px 17px;
    min-height: 110px;
    box-shadow: 0 3px 10px rgba(30, 56, 100, .035);
}

[data-testid="stMetricLabel"] {
    color: #647895;
    font-size: 12px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #122C55;
    font-size: 25px;
    font-weight: 800;
}

/* Dashboard panels */
.panel-title {
    color: #142D56;
    font-size: 15px;
    font-weight: 750;
    margin: 5px 0 8px;
}

.panel-subtitle {
    color: #7888A2;
    font-size: 11px;
    margin-bottom: 10px;
}

[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid #E6EBF5;
    border-radius: 14px;
    padding: 6px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E6EBF5;
    border-radius: 12px;
    overflow: hidden;
    background: white;
}

.stButton button,
.stDownloadButton button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 39px;
}

.stDownloadButton button {
    background: linear-gradient(100deg, #5165F5, #8854E9);
    color: white;
    border: none;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 9px;
}

div[data-testid="stTabs"] button {
    font-size: 12px;
    font-weight: 600;
}

.footer {
    color: #8390A7;
    font-size: 11px;
    text-align: center;
    padding: 15px 0 0;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def money_to_numeric(series):
    """Convert strings like $1,250.50 into numeric values."""
    return pd.to_numeric(
        series.astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def load_data(upload):
    """Load and clean the retail dataset."""

    if upload is not None:
        raw = pd.read_csv(upload)
    else:
        file_path = Path(__file__).parent / "online_retail.csv"

        if not file_path.exists():
            return None, (
                "Please upload your CSV in the sidebar or put "
                "online_retail.csv in the same folder as app.py."
            )

        raw = pd.read_csv(file_path)

    raw.columns = [normalize_name(col) for col in raw.columns]

    required = [
        "customerid", "invoicedate", "quantity", "unitprice"
    ]

    missing = [col for col in required if col not in raw.columns]

    if missing:
        return None, (
            f"Missing required columns: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    df = raw.copy()

    # Customer IDs are text, not numeric values.
    df["customerid"] = df["customerid"].astype("string").str.strip()
    df["customerid"] = df["customerid"].replace("", pd.NA)

    # Convert dates and numeric columns.
    df["invoicedate"] = pd.to_datetime(
        df["invoicedate"], errors="coerce"
    )
    df["quantity"] = pd.to_numeric(
        df["quantity"], errors="coerce"
    )
    df["unitprice"] = money_to_numeric(df["unitprice"])

    if "totalprice" in df.columns:
        df["totalprice"] = money_to_numeric(df["totalprice"])

    df = df.dropna(
        subset=["customerid", "invoicedate", "quantity", "unitprice"]
    )

    # Retain positive sales transactions.
    # Negative/zero quantities and prices are excluded.
    df = df[
        (df["quantity"] > 0) &
        (df["unitprice"] > 0)
    ].copy()

    # Exclude cancellation invoices when invoice IDs start with C.
    if "invoiceno" in df.columns:
        invoice_ids = df["invoiceno"].astype("string").str.strip()
        cancelled = invoice_ids.str.upper().str.startswith("C", na=False)
        df = df[~cancelled].copy()

    # Calculate revenue consistently.
    df["revenue"] = df["quantity"] * df["unitprice"]

    if df.empty:
        return None, "No valid sales transactions remain after cleaning."

    return df, None


def style_chart(fig, height=300):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=15, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", color="#3E5272", size=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=10)
        )
    )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#EAF0F8", zeroline=False)

    return fig


def panel_title(title, subtitle=None):
    st.markdown(
        f'<div class="panel-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="panel-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


# =========================================================
# 4. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 4px 14px;">
        <div style="
            width:45px;height:45px;
            display:flex;align-items:center;justify-content:center;
            background:linear-gradient(135deg,#4F78F6,#9560EB);
            border-radius:13px;font-size:24px;
            box-shadow:0 5px 16px rgba(79,120,246,.3);
            margin-bottom:12px;">
            🛍️
        </div>
        <div style="font-size:22px;font-weight:800;color:#FFFFFF;
                    letter-spacing:-.6px;">
            Retail Insights
        </div>
        <div style="font-size:11px;color:#AFC1E5;margin-top:5px;">
            CUSTOMER ANALYTICS PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="font-size:10px;font-weight:800;letter-spacing:1.5px;
                color:#8FA6D5;margin-bottom:8px;">
        WORKSPACE
    </div>
    """, unsafe_allow_html=True)

    page_icons = {
        "Overview": "🏠",
        "Customer Segments": "👥",
        "RFM Analysis": "📊",
        "Customer Explorer": "🔎",
        "Download Results": "📥"
    }

    pages = list(page_icons.keys())

    page = st.radio(
        "NAVIGATION",
        pages,
        format_func=lambda name: f"{page_icons[name]}   {name}",
        index=0,
        key="sidebar_page",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div style="font-size:10px;font-weight:800;letter-spacing:1.5px;
                color:#8FA6D5;margin-bottom:10px;">
        DATA SOURCE
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload retail dataset",
        type=["csv"],
        help="Upload a CSV with CustomerID, InvoiceDate, Quantity and UnitPrice."
    )

    st.caption("CSV files only · Retail transaction data")

    st.divider()

    st.markdown("""
    <div style="font-size:10px;font-weight:800;letter-spacing:1.5px;
                color:#8FA6D5;margin-bottom:8px;">
        MODEL SETTINGS
    </div>
    """, unsafe_allow_html=True)

    k = st.slider(
        "Number of customer segments",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Choose how many customer groups K-Means should create."
    )

    st.markdown(f"""
    <div style="background:#20365D;border:1px solid #3B5078;
                padding:12px;border-radius:11px;margin-top:8px;">
        <div style="font-size:11px;color:#B8C9EB;">
            ACTIVE CLUSTER COUNT
        </div>
        <div style="font-size:26px;font-weight:800;color:white;">
            {k}
        </div>
        <div style="font-size:11px;color:#B8C9EB;">
            K-Means configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="sidebar-note">
        <div style="font-weight:800;font-size:13px;color:white;
                    margin-bottom:8px;">
            💡 RFM QUICK GUIDE
        </div>
        <div><b style="color:#8EB4FF;">R</b> · Recency</div>
        <div style="font-size:11px;">Days since last purchase</div>
        <div style="margin-top:7px;"><b style="color:#6BE3B1;">F</b> · Frequency</div>
        <div style="font-size:11px;">Unique invoice dates</div>
        <div style="margin-top:7px;"><b style="color:#D1B2FF;">M</b> · Monetary</div>
        <div style="font-size:11px;">Total customer revenue</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;color:#8194B9;font-size:10px;
                padding:20px 0 5px;">
        RETAIL INSIGHTS · V1.0
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 5. LOAD AND VALIDATE DATA
# =========================================================
try:
    df, error = load_data(uploaded_file)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if error:
    st.warning(error)
    st.stop()

unique_customer_count = df["customerid"].nunique()

if unique_customer_count < k:
    st.error(
        f"There are only {unique_customer_count} unique customers. "
        "Choose a smaller cluster count."
    )
    st.stop()


# =========================================================
# 6. RFM CALCULATION
# =========================================================
snapshot = df["invoicedate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customerid").agg(
    Recency=(
        "invoicedate",
        lambda dates: (snapshot - dates.max()).days
    ),
    # Required by the original assignment:
    # Frequency is the number of unique invoice dates.
    Frequency=("invoicedate", "nunique"),
    Monetary=("revenue", "sum")
)

rfm.index.name = "CustomerID"
rfm = rfm.reset_index()
rfm["Monetary"] = rfm["Monetary"].round(2)


# =========================================================
# 7. K-MEANS CLUSTERING
# =========================================================
features = ["Recency", "Frequency", "Monetary"]

scaler = StandardScaler()
scaled = scaler.fit_transform(rfm[features])

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(scaled)

rfm["Segment"] = [f"Segment {label + 1}" for label in labels]

segment_order = sorted(
    rfm["Segment"].unique(),
    key=lambda value: int(value.split()[-1])
)

segment_colors = {
    segment: COLORS[i % len(COLORS)]
    for i, segment in enumerate(segment_order)
}

summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean")
    )
    .round(2)
    .reindex(segment_order)
    .reset_index()
)

if 1 < len(set(labels)) < len(rfm):
    silhouette = silhouette_score(scaled, labels)
else:
    silhouette = None

total_customers = len(rfm)
total_transactions = len(df)
total_revenue = df["revenue"].sum()


# =========================================================
# 8. MAIN DASHBOARD HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-label">RETAIL ANALYTICS PLATFORM</div>
    <h1>Online Retail Customer Segmentation</h1>
    <p>
        Discover customer behavior with RFM Analysis and K-Means Clustering
    </p>
</div>
""", unsafe_allow_html=True)

status_col, date_col = st.columns([1, 2])

with status_col:
    st.caption("🟢 Dataset loaded and analyzed")

with date_col:
    st.caption(
        f"📅 Latest transaction: "
        f"{df['invoicedate'].max():%d %b %Y %H:%M}"
    )


# =========================================================
# 9. KPI CARDS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("👥 Total Customers", f"{total_customers:,}")
c2.metric("▤ Transactions", f"{total_transactions:,}")
c3.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
c4.metric("▦ Customer Segments", str(k))


# =========================================================
# 10. OVERVIEW DASHBOARD
# =========================================================
if page == "Overview":

    left, middle, right = st.columns(
        [1.12, 1.08, 0.78],
        gap="small"
    )

    with left:
        panel_title("📊 Customer Distribution by Segment")

        fig = px.bar(
            summary,
            x="Segment",
            y="Customers",
            color="Segment",
            text="Customers",
            color_discrete_map=segment_colors,
            category_orders={"Segment": segment_order}
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(
            style_chart(fig, 315),
            use_container_width=True
        )

    with middle:
        panel_title("📋 Average RFM Values by Segment")

        table = summary.rename(columns={
            "Recency": "Recency (days)",
            "Frequency": "Frequency",
            "Monetary": "Monetary ($)"
        })

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=315,
            column_config={
                "Customers": st.column_config.NumberColumn(
                    "Customers", format="%d"
                ),
                "Recency (days)": st.column_config.NumberColumn(
                    "Recency (days)", format="%.2f"
                ),
                "Frequency": st.column_config.NumberColumn(
                    "Frequency", format="%.2f"
                ),
                "Monetary ($)": st.column_config.NumberColumn(
                    "Monetary ($)", format="$%.2f"
                )
            }
        )

    with right:
        panel_title("🎯 Model Performance")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=silhouette if silhouette is not None else 0,
            number={
                "valueformat": ".2f",
                "font": {"size": 28, "color": "#142B50"}
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [-1, 1],
                    "tickvals": [-1, 0, 1],
                    "tickfont": {"size": 9}
                },
                "bar": {
                    "color": "#20C6A4",
                    "thickness": 0.26
                },
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [-1, 0], "color": "#FBE6E9"},
                    {"range": [0, 0.5], "color": "#E9F4FF"},
                    {"range": [0.5, 1], "color": "#D9F7EF"}
                ]
            }
        ))

        gauge.update_layout(
            height=235,
            margin=dict(l=15, r=15, t=20, b=10),
            paper_bgcolor="white",
            font=dict(family="Inter")
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.caption(
            "Higher silhouette scores generally indicate better-separated "
            "clusters. This is not an accuracy percentage."
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    scatter_col, detail_col = st.columns([1.02, 1.08], gap="small")

    with scatter_col:
        panel_title("👥 Customer Segments — Recency vs Monetary")

        scatter = px.scatter(
            rfm,
            x="Recency",
            y="Monetary",
            color="Segment",
            size="Frequency",
            hover_data=["CustomerID", "Frequency"],
            color_discrete_map=segment_colors,
            category_orders={"Segment": segment_order},
            labels={
                "Recency": "Recency (days)",
                "Monetary": "Monetary Value ($)",
                "Frequency": "Frequency"
            }
        )

        st.plotly_chart(
            style_chart(scatter, 375),
            use_container_width=True
        )

    with detail_col:
        panel_title("🔎 Customer Intelligence")

        tab_customers, tab_segments, tab_download = st.tabs([
            "Customer Details",
            "Segment Overview",
            "Download"
        ])

        with tab_customers:
            search = st.text_input(
                "Search customer ID",
                placeholder="Enter customer ID...",
                key="overview_search"
            )

            segment_filter = st.selectbox(
                "Customer segment",
                ["All Segments"] + segment_order,
                key="overview_segment_filter"
            )

            customer_data = rfm.copy()

            if search.strip():
                customer_data = customer_data[
                    customer_data["CustomerID"].astype(str).str.contains(
                        search.strip(),
                        case=False,
                        na=False
                    )
                ]

            if segment_filter != "All Segments":
                customer_data = customer_data[
                    customer_data["Segment"] == segment_filter
                ]

            customer_data = customer_data.sort_values(
                "Monetary",
                ascending=False
            )

            st.dataframe(
                customer_data,
                use_container_width=True,
                hide_index=True,
                height=250,
                column_config={
                    "CustomerID": st.column_config.TextColumn(
                        "Customer ID"
                    ),
                    "Recency": st.column_config.NumberColumn(
                        "Recency (days)", format="%d"
                    ),
                    "Frequency": st.column_config.NumberColumn(
                        "Frequency", format="%d"
                    ),
                    "Monetary": st.column_config.NumberColumn(
                        "Monetary ($)", format="$%.2f"
                    )
                }
            )

            st.caption(
                f"Showing {len(customer_data)} matching customers"
            )

        with tab_segments:
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Monetary": st.column_config.NumberColumn(
                        "Monetary ($)", format="$%.2f"
                    )
                }
            )

            chosen_segment = st.selectbox(
                "Inspect a segment",
                segment_order,
                key="segment_profile"
            )

            profile = summary[
                summary["Segment"] == chosen_segment
            ].iloc[0]

            st.markdown(f"**{chosen_segment} profile**")
            st.write(f"Customers: {int(profile['Customers'])}")
            st.write(f"Average recency: {profile['Recency']:.1f} days")
            st.write(f"Average frequency: {profile['Frequency']:.1f}")
            st.write(f"Average monetary: ${profile['Monetary']:,.2f}")

        with tab_download:
            st.write("Download your customer-level results.")

            st.download_button(
                "⬇️ Download Customer CSV",
                data=rfm.to_csv(index=False).encode("utf-8"),
                file_name="customer_segmentation_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Segment Summary",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="customer_segment_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("""
    <div style="background:linear-gradient(100deg,#F0EAFE,#FFFFFF);
                border:1px solid #E8DDFE;border-radius:12px;
                padding:14px 18px;margin-top:10px;">
        <b style="color:#5235A8;">✨ Data Quality Note</b><br>
        <span style="color:#657494;font-size:12px;">
        Check your dataset documentation for cancellation and return
        conventions before presenting final business findings.
        </span>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 11. CUSTOMER SEGMENTS PAGE
# =========================================================
elif page == "Customer Segments":

    panel_title("Customer Segment Analysis")

    chosen = st.selectbox(
        "Choose a segment",
        ["All Segments"] + segment_order
    )

    data = rfm.copy()

    if chosen != "All Segments":
        data = data[data["Segment"] == chosen]

    a, b, c = st.columns(3)

    a.metric("Customers", f"{len(data):,}")

    b.metric(
        "Average Recency",
        f"{data['Recency'].mean():.1f} days"
        if not data.empty else "N/A"
    )

    c.metric(
        "Average Monetary",
        f"${data['Monetary'].mean():,.2f}"
        if not data.empty else "N/A"
    )

    fig = px.bar(
        summary,
        x="Segment",
        y="Monetary",
        color="Segment",
        color_discrete_map=segment_colors,
        labels={"Monetary": "Average Monetary ($)"}
    )

    st.plotly_chart(
        style_chart(fig, 380),
        use_container_width=True
    )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 12. RFM ANALYSIS PAGE
# =========================================================
elif page == "RFM Analysis":

    panel_title("RFM Analysis")

    st.markdown("""
    **Recency:** Days since the latest purchase. Lower is more recent.

    **Frequency:** Number of unique InvoiceDate values per customer.

    **Monetary:** Total revenue from retained transactions.
    """)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            rfm,
            x="Recency",
            nbins=20,
            color_discrete_sequence=["#3980F6"],
            labels={"Recency": "Recency (days)"}
        )

        st.plotly_chart(
            style_chart(fig, 320),
            use_container_width=True
        )

    with c2:
        fig = px.histogram(
            rfm,
            x="Frequency",
            nbins=20,
            color_discrete_sequence=["#9854E8"],
            labels={"Frequency": "Unique Invoice Dates"}
        )

        st.plotly_chart(
            style_chart(fig, 320),
            use_container_width=True
        )

    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        size="Frequency",
        color="Segment",
        color_discrete_map=segment_colors,
        hover_data=["CustomerID"]
    )

    st.plotly_chart(
        style_chart(fig, 420),
        use_container_width=True
    )

    st.markdown("#### Full Customer RFM Table")

    st.dataframe(
        rfm.sort_values("Monetary", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 13. CUSTOMER EXPLORER PAGE
# =========================================================
elif page == "Customer Explorer":

    panel_title("Customer Explorer")

    c1, c2, c3 = st.columns([1.4, 1, 1])

    with c1:
        search = st.text_input(
            "Search Customer ID",
            placeholder="Enter customer ID..."
        )

    with c2:
        seg = st.selectbox(
            "Filter Segment",
            ["All Segments"] + segment_order
        )

    with c3:
        sort = st.selectbox(
            "Sort By",
            ["Monetary", "Recency", "Frequency"]
        )

    data = rfm.copy()

    if search.strip():
        data = data[
            data["CustomerID"].astype(str).str.contains(
                search.strip(),
                case=False,
                na=False
            )
        ]

    if seg != "All Segments":
        data = data[data["Segment"] == seg]

    data = data.sort_values(
        sort,
        ascending=(sort == "Recency")
    )

    st.caption(f"{len(data):,} customers found")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    if not data.empty:
        selected = st.selectbox(
            "View customer profile",
            data["CustomerID"].astype(str).tolist()
        )

        row = data[
            data["CustomerID"].astype(str) == selected
        ].iloc[0]

        a, b, c, d = st.columns(4)

        a.metric("Customer ID", str(row["CustomerID"]))
        b.metric("Recency", f"{row['Recency']} days")
        c.metric("Frequency", int(row["Frequency"]))
        d.metric("Monetary", f"${row['Monetary']:,.2f}")

        st.info(f"Assigned cluster: {row['Segment']}")


# =========================================================
# 14. DOWNLOAD RESULTS PAGE
# =========================================================
elif page == "Download Results":

    panel_title("Download Your Analysis")

    st.write(
        "Export your customer-level RFM results and segment summary "
        "for Excel, reports, or presentations."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 👥 Customer RFM Results")
        st.write(
            "Customer ID, Recency, Frequency, Monetary, and segment."
        )

        st.download_button(
            "⬇️ Download Customer Results",
            data=rfm.to_csv(index=False).encode("utf-8"),
            file_name="customer_segmentation_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c2:
        st.markdown("#### 📊 Segment Summary")
        st.write(
            "Customer counts and average RFM values per segment."
        )

        st.download_button(
            "⬇️ Download Segment Summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="customer_segment_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("#### Model Information")

    info1, info2, info3 = st.columns(3)

    info1.metric("Algorithm", "K-Means")
    info2.metric("Number of Clusters", k)

    info3.metric(
        "Silhouette Score",
        f"{silhouette:.3f}" if silhouette is not None else "N/A"
    )

    st.caption(
        "The silhouette score measures cluster separation; "
        "it is not an accuracy percentage."
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 15. FOOTER
# =========================================================
st.markdown("---")

st.markdown("""
<div class="footer">
    <b>Retail Insights</b><br>
    Online Retail Customer Segmentation · RFM Analysis · K-Means Clustering
</div>
""", unsafe_allow_html=True)
