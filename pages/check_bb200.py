# Streamlit 기반 BB200 상단 돌파 스크리닝 앱 (입력 조건 확장)
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from mplfinance.original_flavor import candlestick_ohlc
import io

# 한글 폰트 설정
import matplotlib
if platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
else:
    matplotlib.rcParams['font.family'] = 'sans-serif'

#@st.cache_data
def load_krxdata(parquet_path):
    df = pd.read_parquet(parquet_path)
    df.dropna(subset=["Open","High","Low","Close","Volume"], inplace=True)
    return df[df["Volume"] > 0].copy()

def check_market_cap(gdf, threshold):
    return gdf.iloc[-1]["MarketCap"] >= threshold

def check_volume_increase(subdf):
    if len(subdf) < 2:
        return False
    return subdf.iloc[-1]["Volume"] > subdf.iloc[-2]["Volume"]

def check_boll_condition(subdf, vol_ratio_thres, breakout_thres):
    if len(subdf) < 2:
        return False, 0, 0
    idx_last = len(subdf) - 1
    idx_prev = idx_last - 1
    close_prev = subdf.iloc[idx_prev]["Close"]
    bbu_prev = subdf.iloc[idx_prev]["BBAND_U_200"]
    open_today = subdf.iloc[idx_last]["Open"]
    close_today = subdf.iloc[idx_last]["Close"]
    high_today = subdf.iloc[idx_last]["High"]
    low_today = subdf.iloc[idx_last]["Low"]
    bbu_today = subdf.iloc[idx_last]["BBAND_U_200"]
    vol_yesterday = subdf.iloc[idx_prev]["Volume"]
    vol_today = subdf.iloc[idx_last]["Volume"]
    vol_ratio = vol_today / vol_yesterday if vol_yesterday > 0 else 0
    breakout_rise = ((close_today / open_today) - 1) * 100
    oc_bbu200 = (close_today - open_today) / (bbu_today - open_today) if (bbu_today - open_today) != 0 else 0
    lc_hl = (close_today - low_today) / (high_today - low_today) if (high_today - low_today) != 0 else 0
    cond1 = (close_prev < bbu_prev) and (close_today > bbu_today)
    cond2 = (close_today > bbu_today) and (low_today < bbu_today)
    cond3 = (vol_ratio >= vol_ratio_thres) and (oc_bbu200 >= 1.33) and (lc_hl >= 0.5) and (breakout_rise >= breakout_thres)
    return (cond1 or cond2) and cond3, vol_ratio, breakout_rise

def screen_stock(gdf, day_offset, refdate, mcap_threshold, vol_ratio_thres, breakout_thres):
    idx_basis = len(gdf) - 1 - day_offset
    if idx_basis < 0:
        return None
    subdf = gdf.iloc[:idx_basis+1].copy()
    if not check_volume_increase(subdf):
        return None
    cond_met, vol_ratio, breakout_rise = check_boll_condition(subdf, vol_ratio_thres, breakout_thres)
    if cond_met:
        row = gdf.iloc[idx_basis]
        dt_ = row["Date"].strftime("%Y-%m-%d")
        if dt_ != refdate:
            return None
        return {
            "Name": row["Name"],
            "Code": row["Code"],
            "Date": dt_,
            "LastClose": row["Close"],
            "McapOk": check_market_cap(gdf, mcap_threshold * 100000000),
            "Vol_Ratio": vol_ratio,
            "Breakout_Rise": breakout_rise,
            "Data": gdf.iloc[:idx_basis+1].copy()
        }
    return None

def plot_chart(sub_df, title, show_bbu200, save_path=None):
    sub_df = sub_df.tail(60).copy()
    sub_df["XIndex"] = np.arange(len(sub_df))
    ohlc = sub_df[["XIndex", "Open", "High", "Low", "Close"]].values
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,
                                   gridspec_kw={'height_ratios':[4,1]},
                                   figsize=(10,8))
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='r', colordown='b')
    if show_bbu200:
        ax1.plot(sub_df["XIndex"], sub_df["BBAND_U_200"], color="orange")
    for sma, color in zip(["SMA_5","SMA_10","SMA_20","SMA_50"], ["red","orange","limegreen","blue"]):
        if sma in sub_df.columns:
            ax1.plot(sub_df["XIndex"], sub_df[sma], linewidth=0.5, color=color)
    sub_df["VolColor"] = np.where(sub_df["Close"] >= sub_df["Open"], 'r','b')
    ax2.bar(sub_df["XIndex"], sub_df["Volume"], width=0.6, color=sub_df["VolColor"])
    xind = sub_df["XIndex"].values
    dt_labels = pd.to_datetime(sub_df["Date"]).dt.strftime("%Y-%m-%d").values
    ticks = xind[::10]
    labs = dt_labels[::10]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labs, rotation=45, fontsize=8)
    ax1.set_title(title, fontsize=10)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    return fig

# Streamlit 앱 실행 영역
st.set_page_config(page_title="BB200 돌파 스크리닝", layout="wide")
st.markdown("<br><h3>BB200 돌파 스크리닝</h3>", unsafe_allow_html=True)
#st.title("BB200 돌파 스크리닝")
st.markdown("<br>", unsafe_allow_html=True)

parquet_path = "Z. krxdata.parquet"

col1, col2, col3 = st.columns(3)

with col1:
    day_offset = st.number_input("기준일 설정 (n 일전)", min_value=0, max_value=100, value=0)
    st.markdown("<br>", unsafe_allow_html=True)
    #mcap_threshold = st.number_input("시가총액 (억, 이상)", min_value=0, value=3000, step=500)
    mcap_threshold = st.slider("시가총액 (억, 이상)", min_value=0, max_value=10000, value=3000, step=500)
    #vol_ratio_thres = st.number_input("거래량 비율 (배, 이상)", min_value=0, value=2, step=1)
    vol_ratio_thres = st.slider("거래량 비율 (배, 이상) : 전일 거래량 대비 돌파일 거래량 비율", min_value=0, max_value=10, value=2)
    #breakout_thres = st.number_input("돌파 상승률 기준 (%, 이상)", min_value=0, value=5, step=1)
    breakout_thres = st.slider("돌파 상승률 기준 (%, 이상) : 돌파일 시가 대비 종가의 상승률", min_value=1, max_value=10, value=5)
    show_bbu200 = st.checkbox("BB200 라인 표시", value=True)
    save_charts = True

st.markdown("<br>", unsafe_allow_html=True)

if st.button("스크리닝 실행"):
    st.markdown("<br>", unsafe_allow_html=True)
    df = load_krxdata(parquet_path)
    df.sort_values(["Code", "Date"], inplace=True)
    ref_idx = len(df) - 1 - day_offset
    if ref_idx < 0:
        st.error("데이터가 부족합니다.")
    else:
        refdate = df.iloc[ref_idx]["Date"].strftime("%Y-%m-%d")
        grouped = df.groupby("Code", group_keys=False)
        candidates = []
        if save_charts:
            os.makedirs("Z. BB200+", exist_ok=True)
        for code, gdf in grouped:
            result = screen_stock(gdf.reset_index(drop=True), day_offset, refdate,
                                  mcap_threshold, vol_ratio_thres, breakout_thres)
            if result:
                candidates.append(result)
        st.subheader(f"{refdate}\n스크리닝 결과 : {len(candidates)} 종목")
        for c in candidates:
            title = f"{c['Name']} ({c['Code']})"
            if not c['McapOk']:
                title += " (시총 미달)"
            title += f"  [Q {c['Vol_Ratio']:.1f}x  S {c['Breakout_Rise']:.1f}%]"
            with st.expander(title):
                save_path = None
                if save_charts:
                    fname = f"{c['Date']} {c['Name']}.png".replace("/", "_").replace(" ", "_")
                    save_path = os.path.join("Z. BB200+", fname)
                fig = plot_chart(c['Data'], title, show_bbu200, save_path=save_path)
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
