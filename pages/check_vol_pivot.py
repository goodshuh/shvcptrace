# Streamlit 기반 볼륨 피벗 스크리닝 앱 (조건 입력 확장 + BB200 표시 체크박스)
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from mplfinance.original_flavor import candlestick_ohlc
import io

# 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'

#@st.cache_data
def load_krxdata(path):
    df = pd.read_parquet(path)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return df[df["Volume"] > 0].copy()

def check_market_cap(gdf, threshold):
    return gdf.iloc[-1]["MarketCap"] >= threshold

def check_moving_average_condition(gdf):
    if len(gdf) < 200:
        return False
    sma50 = gdf["Close"].rolling(50).mean().iloc[-1]
    sma100 = gdf["Close"].rolling(100).mean().iloc[-1]
    sma200 = gdf["Close"].rolling(200).mean().iloc[-1]
    return sma50 > sma100 > sma200

def find_peak_info(gdf):
    sub = gdf.iloc[-90:]
    peak_value = sub["Close"].max()
    peak_idx = sub["Close"].idxmax()
    return peak_value, peak_idx

def check_base_2weeks_after_peak(gdf, peak_idx):
    pos = list(gdf.index)
    if peak_idx not in pos:
        return False
    return (len(pos) - 1 - pos.index(peak_idx)) >= 9

def check_40pct_no_break(gdf, peak_value, maxdrop_limit):
    return not (gdf["Low"] < (1 - maxdrop_limit) * peak_value).any()

def check_drop_limit(basis_close, peak_value, threshold):
    return basis_close > 0 and (peak_value - basis_close) / peak_value <= threshold

def check_volume_condition(gdf, n1=5, n2=10):
    return len(gdf) >= n2

def check_volume_pivot(gdf, day_offset=0):
    idx_basis = len(gdf) - 1 - day_offset
    if idx_basis < 90:
        return False, None, None
    base_close = gdf.iloc[idx_basis]["Close"]
    top3 = gdf.iloc[idx_basis-90:idx_basis].nlargest(3, "Volume")
    for _, row in top3.iterrows():
        if row["Close"] < base_close <= row["Close"] * 1.02:
            return True, row["Close"], row["Date"]
    return False, None, None

def screen_candidates(gdf, day_offset, drop_limit, maxdrop_limit):
    if len(gdf) < 60:
        return []
    idx_basis = len(gdf) - 1 - day_offset
    if idx_basis < 1:
        return []
    sub = gdf.iloc[:idx_basis+1].copy()
    peak_val, peak_idx = find_peak_info(sub)
    if not peak_val or peak_val <= 0 or peak_idx is None:
        return []
    if not check_base_2weeks_after_peak(sub, peak_idx):
        return []
    if not check_40pct_no_break(sub.loc[peak_idx:], peak_val, maxdrop_limit):
        return []
    basis_close = sub.iloc[-1]["Close"]
    prev_close = sub.iloc[-2]["Close"]
    if not check_drop_limit(basis_close, peak_val, drop_limit):
        return []
    if not check_volume_condition(sub):
        return []
    ok, pivot_close, pivot_date = check_volume_pivot(sub, day_offset)
    if not ok:
        return []
    if not (prev_close < pivot_close < basis_close <= pivot_close * 1.02):
        return []
    row = sub.iloc[-1]
    return [(row["Code"], row["Name"], row["Date"], basis_close, pivot_close, pivot_date)]

def plot_chart(df, pivot_close, close, name, refdate, pivot_date, show_bbu200=True, save_path=None):
    df = df[df["Date"] <= refdate].copy().tail(90)
    df.reset_index(drop=True, inplace=True)
    df["X"] = np.arange(len(df))
    ohlc = df[["X", "Open", "High", "Low", "Close"]].values
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='r', colordown='b')
    for sma, color in zip(["SMA_5", "SMA_10", "SMA_20", "SMA_50"], ["red", "orange", "lightgreen", "blue"]):
        if sma in df.columns:
            ax1.plot(df["X"], df[sma], color=color, linewidth=0.5)
    if show_bbu200 and "BBAND_U_200" in df.columns:
        ax1.plot(df["X"], df["BBAND_U_200"], color="orange", linewidth=1)
    ax1.axhline(pivot_close, color="magenta", linestyle="--", linewidth=1)
    ax1.text(df["X"].min() + 0.5, pivot_close, f"{pivot_close:,}", fontsize=8, color="magenta", verticalalignment='bottom', horizontalalignment='left')
    
    if pivot_date in df["Date"].values:
        x_pos = df[df["Date"] == pivot_date].index[0]
        ax1.scatter(df.loc[x_pos, "X"], pivot_close, color="magenta", marker="o", s=30, zorder=4)

    top3 = df.nlargest(3, "Volume").sort_values("Date")
    tolerance = 1e-6
    filtered = top3[~np.isclose(top3["Close"], pivot_close, atol=tolerance)]
    for _, row in filtered.iterrows():
        y = row["Close"]
        x = row["X"]
        ax1.axhline(y, color="grey", linestyle="--", linewidth=1)
        ax1.scatter(x, y, color="grey", marker="o", s=30)
        ax1.text(df["X"].min() + 0.5, y, f"{y:,}", fontsize=8, color="grey", verticalalignment='bottom', horizontalalignment='left')
    df["VolColor"] = np.where(df["Close"] >= df["Open"], 'r', 'b')
    ax2.bar(df["X"], df["Volume"], width=0.6, color=df["VolColor"])
    ticks = df["X"].values[::10]
    labels = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").values[::10]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    ax1.set_title(f"{refdate.strftime('%Y-%m-%d')}  {name} (종가: {close:,} | 기준 피벗: {pivot_close:,})", fontsize=10)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    return fig

# Streamlit 앱 시작
st.set_page_config(page_title="볼륨 피벗 돌파 스크리닝", layout="wide")
st.markdown("<br><h3>볼륨 피벗 돌파 스크리닝</h3>", unsafe_allow_html=True)
#st.title("볼륨 피벗 돌파 스크리닝")
st.markdown("<br>", unsafe_allow_html=True)

path = "Z. krxdata.parquet"

col1, col2, col3 = st.columns(3)

with col1:
    day_offset = st.number_input("기준일 설정 (n 일전)", min_value=0, max_value=100, value=0)
    st.markdown("<br>", unsafe_allow_html=True)
    #mcap_threshold = st.number_input("시가총액 (억, 이상)", min_value=0, value=3000)
    mcap_threshold = st.slider("시가총액 (억, 이상)", min_value=0, max_value=10000, value=3000, step=500)
    #drop_limit_pct = st.number_input("전고 대비 기준일 하락률 (%, 이하)", min_value=0, max_value=100, value=21, step=1)
    drop_limit_pct = st.slider("전고 대비 기준일 하락률 (%, 이하)", min_value=0, max_value=100, value=21)
    #maxdrop_limit_pct = st.number_input("전고 이후 최대 하락률 (%, 이하)", min_value=0, max_value=100, value=40, step=1)
    maxdrop_limit_pct = st.slider("전고 이후 최대 하락률 (%, 이하)", min_value=0, max_value=100, value=40)
    show_bbu200 = st.checkbox("BB200 라인 표시", value=True)
    save_chart = True

st.markdown("<br>", unsafe_allow_html=True)

if st.button("스크리닝 실행"):
    st.markdown("<br>", unsafe_allow_html=True)
    df = load_krxdata(path)
    df.sort_values(["Code", "Date"], inplace=True)
    ref_idx = len(df) - 1 - day_offset
    if ref_idx < 0:
        st.error("데이터가 부족합니다.")
    else:
        refdate = df.iloc[ref_idx]["Date"].strftime("%Y-%m-%d")
        grouped = df.groupby("Code", group_keys=False)
        cands = []
        for code, gdf in grouped:
            gdf = gdf.reset_index(drop=True)
            if not check_market_cap(gdf, mcap_threshold * 1e8):
                continue
            if not check_moving_average_condition(gdf):
                continue
            today = screen_candidates(gdf, day_offset, drop_limit_pct / 100, maxdrop_limit_pct / 100)
            if today:
                cands.extend(today)
        st.subheader(f"{refdate}\n스크리닝 결과 : {len(cands)} 종목")
        if save_chart:
            os.makedirs("Z. VOL PV+", exist_ok=True)
        for cd, nm, dt, lcls, pivot, pv_date in cands:
            sub = df[df["Code"] == cd].reset_index(drop=True)
            chart_data = sub[sub["Date"] <= dt].copy().tail(90)
            if chart_data.empty or len(chart_data) < 2:
                chart_data = sub.tail(90).copy()
            title = f"{nm} ({cd})"
            with st.expander(title):
                fname = f"{dt.strftime('%Y-%m-%d')} {nm}.png".replace("/", "_")
                save_path = os.path.join("Z. VOL PV+", fname) if save_chart else None
                fig = plot_chart(chart_data, pivot, lcls, nm, dt, pv_date, show_bbu200=show_bbu200, save_path=save_path)
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=200)
                st.download_button(
                    label="차트 다운로드",
                    data=buf.getvalue(),
                    file_name=fname,
                    mime="image/png"
                )
                plt.close(fig)

# BACK 버튼
coll, colr = st.columns([9, 1])
with colr:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="../" target="_self">
        <div style="
            display: block;
            background-color: #007AFF;
            color: white;
            padding: 0.5em 0.75em;
            text-align: center;
            border-radius: 16px;
            text-decoration: none;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">BACK</div>
    </a>
    """,
    unsafe_allow_html=True)