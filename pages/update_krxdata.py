# Streamlit 기반 KRX 데이터 업데이트 앱 (비밀번호 확인 추가)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykrx import stock

PASSWORD = "236135"  # 여기에 직접 암호를 설정

#@st.cache_data
def load_old_daystock(parquet_path="Z. krxdata.parquet"):
    return pd.read_parquet(parquet_path)

def fetch_recent_ohlcv_all(days=7):
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    all_codes = kospi + kosdaq

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)
    end_str, start_str = end_dt.strftime("%Y%m%d"), start_dt.strftime("%Y%m%d")

    results = []
    progress_bar = st.progress(0, text="KRX 데이터 수집 중...")
    total = len(all_codes)

    for i, code in enumerate(all_codes):
        try:
            df_ohlcv = stock.get_market_ohlcv_by_date(start_str, end_str, code)
            df_ohlcv = df_ohlcv.reset_index()
            df_ohlcv.rename(columns={
                "날짜": "Date", "시가": "Open", "고가": "High",
                "저가": "Low", "종가": "Close", "거래량": "Volume"
            }, inplace=True)
            df_ohlcv["Code"] = code
            df_ohlcv.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
            df_ohlcv = df_ohlcv[df_ohlcv["Volume"] > 0]
            if df_ohlcv.empty:
                continue

            df_ohlcv["Name"] = stock.get_market_ticker_name(code)
            cap_df = stock.get_market_cap_by_date(start_str, end_str, code).reset_index()
            cap_df.rename(columns={"날짜": "Date", "시가총액": "MarketCap"}, inplace=True)
            df_merged = pd.merge(df_ohlcv, cap_df[["Date", "MarketCap"]], on="Date", how="left")

            base_cols = ["Date", "Code", "Name", "Open", "High", "Low", "Close", "Volume", "MarketCap"]
            df_merged = df_merged[base_cols]
            results.append(df_merged)
        except:
            continue
        progress_bar.progress((i + 1) / total, text=f"진행 중... {i + 1}/{total} 종목")

    progress_bar.empty()

    if results:
        new_df = pd.concat(results, ignore_index=True)
        new_df.sort_values(["Code", "Date"], inplace=True)
        return new_df.reset_index(drop=True)
    return pd.DataFrame()

def merge_old_and_new(old_df, new_df):
    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged.drop_duplicates(subset=["Code", "Date"], keep="last", inplace=True)
    merged.sort_values(["Code", "Date"], inplace=True)
    return merged.reset_index(drop=True)

def recalc_indicators(df):
    def calc_one(g):
        g = g.sort_values("Date").copy()
        for w in [5, 10, 20, 50, 100, 200]:
            g[f"SMA_{w}"] = g["Close"].rolling(window=w, min_periods=1).mean()
        ma200 = g["Close"].rolling(window=200, min_periods=1).mean()
        std200 = g["Close"].rolling(window=200, min_periods=1).std()
        g["BBAND_U_200"] = ma200 + 2 * std200
        g["BBAND_L_200"] = ma200 - 2 * std200
        return g
    return df.groupby("Code", group_keys=False).apply(calc_one).reset_index(drop=True)

# Streamlit 앱 시작
st.set_page_config(page_title="KRX 데이터 업데이트", layout="wide")
st.markdown("<br><h3>KRX 데이터 업데이트</h3>", unsafe_allow_html=True)
#st.title("KRX 데이터 업데이트")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("SH 전용 기능입니다.<br>", unsafe_allow_html=True)

parquet_path = "Z. krxdata.parquet"

col1, col2, col3 = st.columns(3)

with col1:
    password_input = st.text_input("비밀번호", type="password")

    if password_input == PASSWORD:
        st.markdown("<br>", unsafe_allow_html=True)
        days = st.slider("업데이트 일 수", min_value=2, max_value=30, value=7)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("업데이트 실행"):
            st.markdown("<br>", unsafe_allow_html=True)
            old_df = load_old_daystock(parquet_path)
            if "MarketCap" not in old_df.columns:
                old_df["MarketCap"] = np.nan

            new_df = fetch_recent_ohlcv_all(days)
            if new_df.empty:
                st.warning("새로 가져온 데이터가 없습니다.")
            else:
                merged_df = merge_old_and_new(old_df, new_df)
                merged_df = recalc_indicators(merged_df)

                final_columns = [
                    "Date", "Code", "Name", "Open", "High", "Low", "Close",
                    "Volume", "MarketCap", "SMA_5", "SMA_10", "SMA_20",
                    "SMA_50", "SMA_100", "SMA_200", "BBAND_U_200", "BBAND_L_200"
                ]
                merged_df = merged_df[final_columns]
                merged_df.to_parquet(parquet_path, engine="pyarrow", index=False)

                st.success(f"데이터 업데이트 완료 : 총 {len(merged_df):,} 행")
                st.dataframe(merged_df.tail(10))
    else:
        st.warning("비밀번호를 입력해야 업데이트를 실행할 수 있습니다.")

# BACK 버튼
coll, colr = st.columns([9, 1])
with colr:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="../streamlit_app.py" target="_self">
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
