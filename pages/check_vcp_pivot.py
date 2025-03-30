# Streamlit 기반 VCP 피벗 스크리닝 앱 (전체 조건 유지)
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import platform
import io

if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'

#@st.cache_data
def load_krxdata(parquet_path):
    df = pd.read_parquet(parquet_path)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return df[df["Volume"] > 0].copy()

def get_vol_moving_averages(gdf, last_n=20):
    sub = gdf.iloc[-last_n:] if len(gdf) >= last_n else gdf.copy()
    vol_5 = sub.tail(5)["Volume"].mean() if len(sub) >= 5 else None
    vol_10 = sub.tail(10)["Volume"].mean() if len(sub) >= 10 else None
    vol_20 = sub["Volume"].mean()
    return vol_5, vol_10, vol_20

def check_recent_volumes(gdf):
    vol_5, vol_10, vol_20 = get_vol_moving_averages(gdf, last_n=20)
    if None in (vol_5, vol_10, vol_20):
        return False
    return (vol_5 < vol_10 < vol_20)

def find_50day_peak_info(gdf):
    lookback = min(len(gdf), 50)
    sub = gdf.iloc[-lookback:]
    peak_value = sub["Close"].max()
    peak_idx = sub["Close"].idxmax()
    return peak_value, peak_idx

def check_base_2weeks_after_peak(gdf, peak_idx):
    positions = list(gdf.index)
    if peak_idx not in positions:
        return False
    peak_pos = positions.index(peak_idx)
    return (len(positions) - 1 - peak_pos) >= 9

def check_40pct_no_break(gdf, peak_value, maxdrop_limit):
    return not (gdf["Low"] < (1 - maxdrop_limit) * peak_value).any()

def compute_zone_by_threshold(gdf, threshold):
    n = len(gdf)
    if n < 2:
        return None, 0
    i = n - 1
    low_in_zone = gdf.iloc[i]["Low"]
    high_in_zone = gdf.iloc[i]["High"]
    j = i - 1
    while j >= 0:
        low_val = gdf.iloc[j]["Low"]
        high_val = gdf.iloc[j]["High"]
        if low_val < low_in_zone:
            low_in_zone = low_val
        if high_val > high_in_zone:
            high_in_zone = high_val
        ratio = high_in_zone / low_in_zone if low_in_zone > 0 else 9999
        if ratio > threshold:
            break
        j -= 1
    sub = gdf.iloc[j+1 : i+1]
    zone_len = len(sub)
    pivot = sub["High"].max()
    return pivot, zone_len

def compute_two_zones_and_pivots(gdf):
    lookback = min(len(gdf), 50)
    sub_50 = gdf.iloc[-lookback:]
    p_max = sub_50["High"].max()
    p_min = sub_50["Low"].min()
    ratio_50 = p_max / p_min if p_min > 0 else 9999
    allowed = 1.0 + ((ratio_50 - 1.0) / 3.0)
    threshold1 = min(allowed, 1.065)
    lpivot, zlen1 = compute_zone_by_threshold(gdf, threshold1)
    threshold2 = min(allowed, 1.045)
    tpivot, zlen2 = compute_zone_by_threshold(gdf, threshold2)
    return lpivot, zlen1, tpivot, zlen2

def is_priority_by_bigvol_close(gdf, last_close, lookback=50):
    sub = gdf.tail(lookback)
    if len(sub) < 1:
        return False
    sub_sorted = sub.sort_values("Volume", ascending=False)
    top1_close = sub_sorted.iloc[0]["Close"]
    top2_close = sub_sorted.iloc[1]["Close"] if len(sub_sorted) > 1 else None
    def within_2pct(x):
        return 0 < ((last_close - x) / x) <= 0.02
    return within_2pct(top1_close) or (top2_close is not None and within_2pct(top2_close))

def screen_candidates(gdf_, day_offset, maxdrop_limit):
    res_list = []
    if len(gdf_) < 60:
        return res_list
    idx_basis = len(gdf_) - 1 - day_offset
    if idx_basis < 0:
        return res_list
    subdf = gdf_.iloc[:idx_basis+1].copy()
    peak_val, peak_idx = find_50day_peak_info(subdf)
    if not peak_val or peak_val <= 0 or peak_idx is None:
        return res_list
    if not check_base_2weeks_after_peak(subdf, peak_idx):
        return res_list
    sub_after = subdf.loc[peak_idx:]
    if not check_40pct_no_break(sub_after, peak_val, maxdrop_limit):
        return res_list
    basis_row = subdf.iloc[-1]
    basis_close = basis_row["Close"]
    if basis_close <= 0 or ((peak_val - basis_close) / peak_val) > 0.27:
        return res_list
    sma50 = basis_row.get("SMA_50")
    sma100 = basis_row.get("SMA_100")
    sma200 = basis_row.get("SMA_200")
    if not (sma50 and sma100 and sma200 and (sma50 > sma100 > sma200) and (basis_close > sma50)):
        return res_list
    high_basis = basis_row["High"]
    prev_close = subdf.iloc[-2]["Close"]
    if prev_close <= 0 or ((high_basis - prev_close) / prev_close) > 0.04:
        return res_list
    lpivot, zlen1, tpivot, zlen2 = compute_two_zones_and_pivots(subdf)
    if not lpivot or lpivot <= 0:
        return res_list
    if (basis_close - lpivot) / lpivot > 0.03 or zlen1 < 3:
        return res_list
    if zlen2 < 3:
        tpivot = None
        zlen2 = 0
    if not check_recent_volumes(subdf):
        return res_list
    nm = basis_row["Name"]
    cd = basis_row["Code"]
    dt_ = basis_row["Date"]
    prio = is_priority_by_bigvol_close(subdf, basis_close)
    res_list.append((cd, nm, dt_, basis_close, lpivot, zlen1, tpivot, zlen2, prio))
    return res_list

# Streamlit 앱 시작
st.set_page_config(page_title="VCP 피벗 후보 스크리닝", layout="wide")
st.markdown("<br><h3>VCP 피벗 후보 스크리닝</h3>", unsafe_allow_html=True)
#st.title("VCP 피벗 스크리닝")
st.markdown("<br>", unsafe_allow_html=True)

path = "Z. krxdata.parquet"
save_folder = "Z. VCP PV-"

col1, col2, col3 = st.columns(3)

with col1:
    day_offset = st.number_input("기준일 설정 (n 일 전)", min_value=0, max_value=100, value=0)
    st.markdown("<br>", unsafe_allow_html=True)
    #mcap_threshold = st.number_input("시가총액 기준 (억 이상)", min_value=0, value=3000)
    mcap_threshold = st.slider("시가총액 (억, 이상)", min_value=0, max_value=10000, value=3000, step=500)
    #maxdrop_limit = st.number_input("전고 이후 최대 하락률 (%, 이하)", min_value=0, max_value=100, value=40, step=1)
    maxdrop_limit = st.slider("전고 이후 최대 하락률 (%, 이하)", min_value=0, max_value=100, value=40)
    show_bbu200 = st.checkbox("BB200 라인 표시", value=True)
    save_chart = True

st.markdown("<br>", unsafe_allow_html=True)

if st.button("스크리닝 실행"):
    st.markdown("<br>", unsafe_allow_html=True)    
    df = load_krxdata(path)
    df.sort_values(["Code", "Date"], inplace=True)
    ref_idx = len(df) - 1 - day_offset
    refdate = df.iloc[ref_idx]["Date"].strftime("%Y-%m-%d") if ref_idx >= 0 else "noData"
    grouped = df.groupby("Code", group_keys=False)
    all_candidates = []
    for code, gdf in grouped:
        gdf = gdf.reset_index(drop=True)
        if gdf.iloc[-1]["MarketCap"] < mcap_threshold * 1e8:
            continue
        today_cands = screen_candidates(gdf, day_offset, maxdrop_limit / 100)
        if today_cands:
            all_candidates.extend(today_cands)
    st.subheader(f"{refdate}\n스크리닝 결과 : {len(all_candidates)} 종목")

    # 후보 리스트 CSV 저장
    save_csv = "Z. list_vcppivot.csv"
    if all_candidates:
        recs = []
        for cand in all_candidates:
            cd, nm, dt_, lcls, p1, z1, p2, z2, _ = cand
            recs.append({
                "Name": nm,
                "Code": cd,
                "RegDate": dt_.strftime("%Y-%m-%d") if isinstance(dt_, pd.Timestamp) else str(dt_),
                "LastClose": lcls,
                "Loose PV": p1,
                "ZLen1": z1,
                "Tight PV": p2 if p2 and p2 > 0 and z2 >= 3 else None,
                "ZLen2": z2 if p2 and p2 > 0 and z2 >= 3 else None
            })
        newdf = pd.DataFrame(recs, columns=["Name", "Code", "RegDate", "LastClose", "Loose PV", "ZLen1", "Tight PV", "ZLen2"])
        newdf["Code"] = newdf["Code"].astype(str).str.zfill(6)
        if os.path.exists(save_csv):
            olddf = pd.read_csv(save_csv, dtype={"Code": str})
            merged = pd.concat([olddf, newdf], ignore_index=True)
            merged.drop_duplicates(subset=["Name", "RegDate"], keep="last", inplace=True)
            merged.sort_values(["Name", "RegDate"], inplace=True)
            merged.to_csv(save_csv, index=False)
            st.info("리스트 갱신 완료")
        else:
            newdf.sort_values(["Name", "RegDate"], inplace=True)
            newdf.to_csv(save_csv, index=False)
            st.info("리스트 생성 완료")

    if save_chart:
        os.makedirs(save_folder, exist_ok=True)

    for cand in all_candidates:
        cd, nm, dt_, lcls, p1, z1, p2, z2, prio = cand
        sub = df[df["Code"] == cd].reset_index(drop=True)
        idx = len(sub) - 1 - day_offset
        if idx < 0:
            continue
        sub_basis = sub.iloc[:idx+1].copy().tail(60)
        sub_basis["XIndex"] = np.arange(len(sub_basis))
        ohlc = sub_basis[["XIndex", "Open", "High", "Low", "Close"]].values
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='r', colordown='b')

        if p1 and p1 > 0:
            ax1.axhline(y=p1, color="green", linestyle="--", linewidth=1, label="Loose PV")
            ax1.text(sub_basis["XIndex"].min() + 0.5, p1, f"{p1:,}", fontsize=8, color="green", verticalalignment='bottom', horizontalalignment='left')
        if p2 and p2 > 0 and z2 >= 3:
            ax1.axhline(y=p2, color="magenta", linestyle="--", linewidth=1, label="Tight PV")
            ax1.text(sub_basis["XIndex"].min() + 0.5, p2, f"{p2:,}", fontsize=8, color="magenta", verticalalignment='bottom', horizontalalignment='left')
        if show_bbu200 and "BBAND_U_200" in sub_basis.columns:
            ax1.plot(sub_basis["XIndex"], sub_basis["BBAND_U_200"], color="orange")
        for sma, color in zip(["SMA_5", "SMA_10", "SMA_20", "SMA_50"], ["red", "orange", "lightgreen", "blue"]):
            if sma in sub_basis.columns:
                ax1.plot(sub_basis["XIndex"], sub_basis[sma], color=color, linewidth=0.5)

        sub_basis["VolColor"] = np.where(sub_basis["Close"] >= sub_basis["Open"], 'r', 'b')
        ax2.bar(sub_basis["XIndex"], sub_basis["Volume"], width=0.6, color=sub_basis["VolColor"])
        xind = sub_basis["XIndex"].values
        dt_labels = pd.to_datetime(sub_basis["Date"]).dt.strftime("%Y-%m-%d").values
        ticks = xind[::10]
        labs = dt_labels[::10]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labs, rotation=45, fontsize=8)
        for label in ax1.get_yticklabels():
            label.set_fontsize(8)
        for label in ax2.get_yticklabels():
            label.set_fontsize(8)

        title = f"{dt_.strftime('%Y-%m-%d')}  {nm}{'**' if prio else ''} (종가 : {lcls:,} | Loose PV : {p1:,}"
        if p2 and p2 > 0 and z2 >= 3:
            title += f" | Tight PV : {p2:,}"
        title += ")"
        ax1.set_title(title, fontsize=10)

        with st.expander(title):
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            fname = f"{dt_.strftime('%Y-%m-%d')} {nm}".replace(" ", "_").replace("/", "_") + ".png"
            if save_chart:
                with open(os.path.join(save_folder, fname), "wb") as f:
                    f.write(buf.getvalue())
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
