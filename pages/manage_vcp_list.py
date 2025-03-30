# Streamlit 기반 VCP 후보 종목 추적 앱 (등록일 세로선 포함)
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import platform
import io

# 폰트 설정
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'

# 파일 경로 설정
CSV_FILE = "Z. list_vcppivot.csv"
PARQUET_FILE = "Z. krxdata.parquet"
PASSWORD = "236135"

#@st.cache_data
def load_krx_data(path):
    df = pd.read_parquet(path)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    df = df[df["Volume"] > 0].copy()
    return df

#@st.cache_data
def load_candidate_list(csv_path):
    df = pd.read_csv(csv_path, dtype={"Code": str})
    df.sort_values(["Name", "RegDate"], inplace=True)
    return df.groupby("Code", as_index=False).first()

def plot_chart(sub_df, pivots, name, save_path=None):
    sub_df = sub_df.tail(60).copy()
    sub_df["XIndex"] = np.arange(len(sub_df))
    ohlc = sub_df[["XIndex", "Open", "High", "Low", "Close"]].values

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='r', colordown='b')

    if pivots.get("Loose PV"):
        lpv = pivots["Loose PV"]
        ax1.axhline(y=lpv, color="green", linestyle="--", linewidth=1, label="Loose PV")
        ax1.text(sub_df["XIndex"].min()+0.5, lpv, f"{lpv:,}", fontsize=8, color="green", verticalalignment='bottom', horizontalalignment='left')
    if pivots.get("Tight PV"):
        tpv = pivots["Tight PV"]
        ax1.axhline(y=tpv, color="magenta", linestyle="--", linewidth=1, label="Tight PV")
        ax1.text(sub_df["XIndex"].min()+0.5, tpv, f"{tpv:,}", fontsize=8, color="magenta", verticalalignment='bottom', horizontalalignment='left')

    if "RegDate" in pivots:
        reg_date = pd.to_datetime(pivots["RegDate"])
        sub_df["Date_dt"] = pd.to_datetime(sub_df["Date"])
        if reg_date in sub_df["Date_dt"].values:
            reg_idx = sub_df[sub_df["Date_dt"] == reg_date].index[0]
            reg_x = sub_df.loc[reg_idx, "XIndex"]
            ax1.axvline(x=reg_x, color="gray", linestyle="dotted", linewidth=1)

    for sma, color in zip(["SMA_5", "SMA_10", "SMA_20", "SMA_50"], ["red", "orange", "lightgreen", "blue"]):
        if sma in sub_df.columns:
            ax1.plot(sub_df["XIndex"], sub_df[sma], color=color, linewidth=0.5)
    if "BBAND_U_200" in sub_df.columns:
        ax1.plot(sub_df["XIndex"], sub_df["BBAND_U_200"], color="orange", linewidth=1)

    sub_df["VolColor"] = np.where(sub_df["Close"] >= sub_df["Open"], 'r', 'b')
    ax2.bar(sub_df["XIndex"], sub_df["Volume"], width=0.6, color=sub_df["VolColor"])

    xind = sub_df["XIndex"].values
    dt_labels = pd.to_datetime(sub_df["Date"]).dt.strftime("%Y-%m-%d").values
    ticks = xind[::10]
    labs = dt_labels[::10]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labs, rotation=45, fontsize=8)

    ax1.set_title(f"{name} 추적 차트", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    return fig

# Streamlit UI
st.set_page_config(page_title="VCP 후보 관리", layout="wide")
st.markdown("<br><h3>VCP 후보 관리</h3>", unsafe_allow_html=True)
#st.title("VCP 후보 종목 추적 차트")

if not os.path.exists(CSV_FILE) or not os.path.exists(PARQUET_FILE):
    st.error("CSV 파일 또는 KRX 데이터 파일이 없습니다.")
    st.stop()

cands_df = load_candidate_list(CSV_FILE)
cands_df["RegDate"] = pd.to_datetime(cands_df["RegDate"])
cands_df = cands_df.sort_values("RegDate", ascending=False).reset_index(drop=True)
df_krx = load_krx_data(PARQUET_FILE)

st.markdown("<BR>", unsafe_allow_html=True)
st.markdown(f"총 {len(cands_df)} 종목 추적중")
st.markdown("<BR>", unsafe_allow_html=True)

# 상승한 종목만 필터링 여부 설정
show_only_breakouts = st.checkbox("피벗 돌파 종목만 보기", value=False)
#show_only_gainers = st.checkbox("등록일 이후 상승한 종목만 보기", value=False)

st.markdown("<BR>", unsafe_allow_html=True)
selected = cands_df["Name"].tolist()#, default=cands_df["Name"].tolist()

for _, row in cands_df.iterrows():
#    if show_only_gainers:
#        reg_price_row = df_krx[(df_krx["Code"] == row["Code"]) & (pd.to_datetime(df_krx["Date"]) == pd.to_datetime(row["RegDate"]))]
#        if not reg_price_row.empty:
#            reg_close = reg_price_row.iloc[0]["Close"]
#            latest_close = df_krx[df_krx["Code"] == row["Code"]].iloc[-1]["Close"]
#            if latest_close <= reg_close:
#                continue
    if row["Name"] not in selected:
        continue
    sub = df_krx[df_krx["Code"] == row["Code"]].copy()
    if sub.empty:
        continue

    latest_row = sub.iloc[-1]
    today_close = latest_row["Close"]
    p1 = row.get("Loose PV")
    p2 = row.get("Tight PV")

    breakout = []
    if pd.notna(p1) and today_close > p1:
        breakout.append("Loose PV")
    if pd.notna(p2) and today_close > p2:
        breakout.append("Tight PV")

    breakout_info = " / ".join(breakout) if breakout else "-"
    if show_only_breakouts and not breakout:
        continue

    with st.expander(f"{row['Name']} ({row['Code']})  ｜ 종가: {today_close:,} ｜ 돌파: {breakout_info}"):
        fig = plot_chart(sub, row, row["Name"])
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.download_button(
                label="차트 다운로드",
                data=buf.getvalue(),
                file_name=f"{row['Code']}_{row['Name']}.png",
                mime="image/png"
            )

        with col2:
            with st.form(f"delete_form_{row['Code']}"):
                #st.markdown("---")
                #st.markdown(f"🔒 **종목 삭제**: {row['Name']} ({row['Code']})")
                pw = st.text_input("비밀번호", type="password", key=f"pw_{row['Code']}", label_visibility="collapsed")
                submitted = st.form_submit_button("종목 삭제")
                if submitted:
                    if pw == PASSWORD:  # 암호 직접 지정
                        orig = pd.read_csv(CSV_FILE, dtype={"Code": str})
                        updated = orig[orig["Code"] != row["Code"]].copy()
                        updated.to_csv(CSV_FILE, index=False)
                        st.success(f"{row['Name']} ({row['Code']}) 삭제 완료")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("비밀번호가 일치하지 않습니다.")
                file_name=f"{row['Code']}_{row['Name']}.png",
                mime="image/png"
        
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