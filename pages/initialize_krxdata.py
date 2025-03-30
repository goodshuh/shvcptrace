# Streamlit 기반 KRX 데이터 업데이트 앱 (모든 종목, 기술지표 포함, 비밀번호 입력)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykrx import stock

PASSWORD = "236135"  # 암호 직접 입력

#@st.cache_data
def fetch_kospi_kosdaq_codes():
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    return [(code, stock.get_market_ticker_name(code)) for code in kospi + kosdaq]

def get_ohlcv_data(code, start_date, end_date):
    df = stock.get_market_ohlcv_by_date(start_date, end_date, code).reset_index()
    df.rename(columns={"날짜": "Date", "시가": "Open", "고가": "High", "저가": "Low",
                       "종가": "Close", "거래량": "Volume"}, inplace=True)
    df["Code"] = code
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    df = df[df["Volume"] > 0].copy()

    cap_df = stock.get_market_cap_by_date(start_date, end_date, code).reset_index()
    cap_df.rename(columns={"날짜": "Date", "시가총액": "MarketCap"}, inplace=True)

    return pd.merge(df, cap_df[["Date", "MarketCap"]], on="Date", how="left")

def add_technical_indicators(df):
    df = df.sort_values("Date").copy()
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
    ma200 = df["Close"].rolling(200, min_periods=1).mean()
    std200 = df["Close"].rolling(200, min_periods=1).std()
    df["BBAND_U_200"] = ma200 + 2 * std200
    df["BBAND_L_200"] = ma200 - 2 * std200
    return df

#def main():
st.set_page_config(page_title="KRX 데이터 초기화", layout="wide")
st.markdown("<br><h3>KRX 데이터 초기화</h3>", unsafe_allow_html=True)
#st.title("KRX 데이터 초기화")
st.markdown("<br>SH 전용 기능입니다.<br>", unsafe_allow_html=True)

path = "Z. krxdata.parquet"
col1, col2, col3 = st.columns(3)

with col1:
    pw = st.text_input("비밀번호", type="password")

    if pw == PASSWORD:
        st.markdown("<br>", unsafe_allow_html=True)
        days = st.slider("초기화 일 수", min_value=300, max_value=2000, value=600, step=100)
        if st.button("초기화 실행"):
            end_date = datetime.today().strftime("%Y%m%d")
            start_date = (datetime.today() - timedelta(days=days * 2)).strftime("%Y%m%d")

            all_info = fetch_kospi_kosdaq_codes()
            progress = st.progress(0, text="데이터 수집 중...")
            results = []

            for i, (code, name) in enumerate(all_info):
                try:
                    df = get_ohlcv_data(code, start_date, end_date)
                    if df.empty:
                        continue
                    df["Name"] = name
                    df = add_technical_indicators(df)
                    df = df[["Date", "Code", "Name", "Open", "High", "Low", "Close", "Volume",
                                "MarketCap", "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200",
                                "BBAND_U_200", "BBAND_L_200"]]
                    results.append(df)
                except Exception as e:
                    print(f"[Error] {code} {name} -> {e}")
                    continue
                progress.progress((i + 1) / len(all_info), text=f"{i + 1}/{len(all_info)} 종목 처리 중")

            progress.empty()
            if not results:
                st.warning("가져온 데이터가 없습니다.")
                #return

            merged_df = pd.concat(results, ignore_index=True)
            merged_df.sort_values(["Code", "Date"], inplace=True)
            merged_df.reset_index(drop=True, inplace=True)
            merged_df.to_parquet(path, engine="pyarrow", index=False)

            st.success(f"데이터 초기화 완료 : 총 {len(merged_df):,} 행")
            st.dataframe(merged_df.tail(10))
    else:
        st.warning("비밀번호를 입력해야 실행할 수 있습니다.")

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

#if __name__ == "__main__":
#    main()
