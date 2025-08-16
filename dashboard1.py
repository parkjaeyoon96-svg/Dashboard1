import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# =========================
# Palette (Green tones)
# =========================
P0 = "#03C55A"  # Deep Green (primary)
P1 = "#4DD27D"  # Medium Green
P2 = "#3BCC73"  # Fresh Green
P3 = "#79DC9C"  # Light Green
P4 = "#C5EED4"  # Very Light Green

COLORWAY = [P0, P1, P2, P3, P4]  # Plotly 순환 색
GRIDCOLOR = "#D9F2E2"            # 격자선 (밝은 그린 톤)
PAPER_BG = "#FFFFFF"             # 차트 바깥
PLOT_BG  = "#F7FCF9"             # 차트 안쪽 (연녹색톤)

# Plotly 템플릿
pio.templates["custom_green"] = go.layout.Template(
    layout=dict(
        colorway=COLORWAY,
        font=dict(family="Pretendard, Noto Sans KR, Segoe UI, Roboto, Arial", size=13, color=P0),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        xaxis=dict(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR),
        yaxis=dict(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR),
        legend=dict(bordercolor="#E6EAF0", borderwidth=0),
        margin=dict(l=20, r=20, t=30, b=30)
    )
)
pio.templates.default = "custom_green"

# =========================
# Streamlit 기본 설정 + 스타일
# =========================
st.set_page_config(page_title="월별 매출 대시보드", layout="wide", page_icon="📊")
st.markdown(
    f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, #f7f9fc 0%, {P4} 100%);
        }}
        .main > div:first-child h1 {{
            color: {P0};
            letter-spacing: 0.2px;
        }}
        .stDivider hr {{
            border-top: 1px solid {GRIDCOLOR};
        }}
        section[data-testid="stSidebar"] {{
            background-color: #f0fdf6 !important;
            border-right: 1px solid {GRIDCOLOR};
        }}
        label, section p {{
            color: {P0} !important;
        }}
        .stButton > button {{
            background: {P0};
            color: white;
            border: 0;
            border-radius: 10px;
        }}
        .stButton > button:hover {{
            background: {P1};
        }}
        .metric-card {{
            background: white;
            border: 1px solid {GRIDCOLOR};
            border-left: 6px solid {P0};
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 4px 10px rgba(3,197,90,0.06);
        }}
        .stDataFrame thead th {{
            background-color: #F2FBF7 !important;
            color: {P0} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 월별 매출 대시보드 (Streamlit)")
st.caption("CSV 업로드 후 4가지 시각화가 자동 생성됩니다. 컬럼: 월(YYYY-MM), 매출액, 전년동월, 증감률(%). 미입력 시 증감률은 전년동월로 자동 계산합니다.")

SAMPLE_CSV = (
    "월,매출액,전년동월,증감률\n"
    "2024-01,12000000,10500000,14.3\n"
    "2024-02,13500000,11200000,20.5\n"
    "2024-03,11000000,12800000,-14.1\n"
    "2024-04,18000000,15200000,18.4\n"
    "2024-05,21000000,18500000,13.5\n"
    "2024-06,22000000,19000000,15.8\n"
    "2024-07,25000000,20500000,22.0\n"
    "2024-08,28000000,24500000,14.3\n"
    "2024-09,24000000,21000000,14.3\n"
    "2024-10,23000000,20000000,15.0\n"
    "2024-11,19500000,17500000,11.4\n"
    "2024-12,17000000,16500000,3.0\n"
)

@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def parse_sample(sample_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(sample_text))

@st.cache_data
def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["월"] = df["월"].astype(str).str.strip()
    df["_date"] = pd.to_datetime(df["월"], format="%Y-%m", errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)
    for c in ["매출액", "전년동월"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "증감률" in df.columns:
        df["증감률"] = pd.to_numeric(df.get("증감률"), errors="coerce")
    else:
        df["증감률"] = np.nan
    missing_mask = df["증감률"].isna()
    df.loc[missing_mask & df["전년동월"].ne(0), "증감률"] = (
        (df.loc[missing_mask, "매출액"] - df.loc[missing_mask, "전년동월"])
        / df.loc[missing_mask, "전년동월"] * 100
    )
    df["증감률"] = df["증감률"].fillna(0)
    df["분기"] = df["_date"].dt.quarter
    return df

# Sidebar
with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader("CSV 업로드", type=["csv"], accept_multiple_files=False)
    use_sample = st.checkbox("샘플 데이터 불러오기", value=True if uploaded is None else False)
    target = st.number_input("KPI 목표 매출 (원)", min_value=0, value=20_000_000, step=100_000)

# Load data
if uploaded is not None:
    df_raw = read_csv(uploaded)
elif use_sample:
    df_raw = parse_sample(SAMPLE_CSV)
else:
    st.info("좌측에서 CSV를 업로드하거나 '샘플 데이터 불러오기'를 선택하세요.")
    st.stop()

# Enrich
try:
    df = enrich_df(df_raw)
except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
    st.stop()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_sales = int(df["매출액"].sum())
    st.markdown(f"<div class='metric-card'><div style='color:{P1};font-size:13px;'>총합 매출</div><div style='color:{P0};font-weight:700;font-size:22px'>{total_sales:,.0f}원</div></div>", unsafe_allow_html=True)
with col2:
    avg_yoy = float(df["증감률"].mean())
    st.markdown(f"<div class='metric-card'><div style='color:{P1};font-size:13px;'>평균 증감률</div><div style='color:{P0};font-weight:700;font-size:22px'>{avg_yoy:.1f}%</div></div>", unsafe_allow_html=True)
with col3:
    max_idx = df["매출액"].idxmax()
    st.markdown(f"<div class='metric-card'><div style='color:{P1};font-size:13px;'>최고 매출 (월)</div><div style='color:{P0};font-weight:700;font-size:22px'>{df.loc[max_idx,'월']} · {df.loc[max_idx,'매출액']:,.0f}원</div></div>", unsafe_allow_html=True)
with col4:
    min_idx = df["매출액"].idxmin()
    st.markdown(f"<div class='metric-card'><div style='color:{P1};font-size:13px;'>최저 매출 (월)</div><div style='color:{P0};font-weight:700;font-size:22px'>{df.loc[min_idx,'월']} · {df.loc[min_idx,'매출액']:,.0f}원</div></div>", unsafe_allow_html=True)

st.divider()

# ===== 2x2 Grid Layout =====
row1_col1, row1_col2 = st.columns(2, gap="large")
row2_col1, row2_col2 = st.columns(2, gap="large")

# 1) 월별 매출 추이 (좌상)
with row1_col1:
    st.subheader("1) 월별 매출 추이 (매출액 vs 전년동월)")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["월"], y=df["매출액"], mode="lines+markers", name="매출액",
        line=dict(width=3, color=P0), marker=dict(size=7, line=dict(width=1, color="#FFFFFF"))
    ))
    fig_trend.add_trace(go.Scatter(
        x=df["월"], y=df["전년동월"], mode="lines+markers", name="전년동월",
        line=dict(width=2, dash="dash", color=P2), marker=dict(size=6)
    ))
    fig_trend.update_layout(yaxis_title="매출액 (원)", xaxis_title="월", height=360)
    st.plotly_chart(fig_trend, use_container_width=True)

# 2) 전년 대비 증감률 (우상)
with row1_col2:
    st.subheader("2) 전년 대비 증감률")
    bar_colors = [P0 if v >= 0 else P2 for v in df["증감률"]]
    fig_yoy = go.Figure(go.Bar(
        x=df["월"], y=df["증감률"], marker_color=bar_colors, name="증감률"
    ))
    fig_yoy.update_layout(yaxis_title="증감률 (%)", xaxis_title="월", height=360)
    st.plotly_chart(fig_yoy, use_container_width=True)

# 3) 분기별 매출 분포 (좌하)
with row2_col1:
    st.subheader("3) 분기별 매출 분포 (Boxplot)")
    fig_box = px.box(df, x="분기", y="매출액", points="all", color_discrete_sequence=[P1])
    fig_box.update_traces(marker=dict(size=6, line=dict(width=1, color="#FFFFFF")))
    fig_box.update_layout(yaxis_title="매출액 (원)", xaxis_title="분기", height=360)
    st.plotly_chart(fig_box, use_container_width=True)

# 4) 월별 KPI 달성률 (우하) + 목표선 빨간 점선
with row2_col2:
    st.subheader("4) 월별 KPI 달성률 (목표선 100%)")
    rate = (df["매출액"] / (target if target else 1)) * 100.0
    fig_kpi = go.Figure()
    fig_kpi.add_trace(go.Scatter(
        x=df["월"], y=rate, mode="lines+markers", name="달성률",
        line=dict(width=3, color=P1), marker=dict(size=7, line=dict(width=1, color="#FFFFFF"))
    ))
    fig_kpi.add_hline(
        y=100, line_dash="dash", line_color="#E53935",
        annotation_text="목표 100%", annotation_position="top left",
        annotation_font=dict(color="#E53935")
    )
    fig_kpi.update_layout(yaxis_title="달성률 (%)", xaxis_title="월", height=360)
    st.plotly_chart(fig_kpi, use_container_width=True)

st.divider()
st.subheader("데이터 미리보기")
st.dataframe(df.drop(columns=["_date"]))

st.caption("Tip: 좌측 사이드바에서 KPI 목표를 바꾸면 달성률 차트가 즉시 반영됩니다. 업로드 파일은 동일 스키마를 유지해주세요.")
