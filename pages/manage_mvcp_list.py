import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import platform
import io

# --------------------------------------
# 초기 세팅
# --------------------------------------
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'

# 파일 경로 설정
CSV_FILE = "Z. list_mvcppivot.csv"
PARQUET_FILE = "Z. krxdata.parquet"
PASSWORD = "236135"

# 세션 상태 초기화
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = False

# --------------------------------------
# (1) 로컬 인덱스 버전의 수축 구간 판별 함수
# --------------------------------------
def find_contraction_intervals(df, peak_lookback=90, min_bars_required=30):
    """
    (로컬 인덱스 버전)
    df는 이미 reset_index(drop=True) 된 상태라고 가정.
    - df.index = 0..len(df)-1
    - peak_lookback만큼 역으로 찾아서 고점 이후 수축 구간을 판별.
    - 반환 intervals = [
        {"start": i_s, "end": i_e, "min_price": ..., "max_price": ...},
        ...
      ]
    """
    if len(df) < min_bars_required:
        return []

    end_i = len(df) - 1
    st_i  = max(0, end_i - peak_lookback + 1)
    sub   = df.iloc[st_i : end_i + 1].copy()
    if sub.empty:
        return []

    sub.reset_index(drop=True, inplace=True)

    peak_local_idx = sub["Close"].idxmax()
    if peak_local_idx >= (len(sub) - 1):
        return []

    cur_start = peak_local_idx + 1
    if cur_start >= len(sub):
        return []

    intervals = []

    def calc_vol(intv):
        lo = intv["min_price"]
        hi = intv["max_price"]
        return (hi / lo - 1.0) if lo > 0 else 0.0

    def finalize_and_merge():
        while len(intervals) >= 2:
            l_ = intervals[-1]
            p_ = intervals[-2]
            v_l = calc_vol(l_)
            v_p = calc_vol(p_)
            cond1 = (v_l > v_p)
            cond2 = (l_["min_price"] < p_["min_price"])
            if cond1 or cond2:
                merged = {
                    "start": p_["start"],
                    "end":   l_["end"],
                    "min_price": min(p_["min_price"], l_["min_price"]),
                    "max_price": max(p_["max_price"], l_["max_price"])
                }
                intervals.pop()
                intervals.pop()
                intervals.append(merged)
            else:
                break

    intervals.append({
        "start": cur_start,
        "end":   cur_start,
        "min_price": float(sub.loc[cur_start, "Low"]),
        "max_price": float(sub.loc[cur_start, "High"])
    })

    for i in range(cur_start+1, len(sub)):
        day_low   = float(sub.loc[i, "Low"])
        day_high  = float(sub.loc[i, "High"])
        day_close = float(sub.loc[i, "Close"])

        c_int = intervals[-1]
        c_int["end"] = i

        current_vol = calc_vol(c_int)

        if day_low < c_int["min_price"]:
            old_min = c_int["min_price"]
            if current_vol <= 0.065:
                if day_close < old_min:
                    c_int["min_price"] = day_low
                else:
                    pass
            else:
                c_int["min_price"] = day_low

        if day_high > c_int["max_price"]:
            if day_close <= c_int["max_price"]:
                c_int["max_price"] = day_high
            else:
                finalize_and_merge()
                intervals.append({
                    "start": i,
                    "end":   i,
                    "min_price": day_low,
                    "max_price": day_high
                })

        if (i - c_int["start"]) >= 2:
            vol_c = calc_vol(c_int)
            sub_idx = list(range(c_int["start"], i+1))
            if len(sub_idx) >= 3:
                last3 = sub_idx[-3:]
                sl_low = sub.loc[last3, "Low"].min()
                sl_hi  = sub.loc[last3, "High"].max()
                rec_vol = (sl_hi / sl_low - 1.0) if sl_low>0 else 0
                if rec_vol <= vol_c * 0.7:
                    finalize_and_merge()
                    intervals.append({
                        "start": i,
                        "end":   i,
                        "min_price": day_low,
                        "max_price": day_high
                    })

    finalize_and_merge()

    out_intervals = []
    for itv in intervals:
        out_intervals.append({
            "start": itv["start"] + st_i,
            "end":   itv["end"]   + st_i,
            "min_price": itv["min_price"],
            "max_price": itv["max_price"]
        })
    return out_intervals


#@st.cache_data
def load_krx_data(path):
    df = pd.read_parquet(path)
    df.dropna(subset=["Open","High","Low","Close","Volume"], inplace=True)
    df = df[df["Volume"]>0].copy()
    return df

#@st.cache_data
def load_candidate_list(csv_path):
    df = pd.read_csv(csv_path, dtype={"Code": str})
    df.sort_values(["Name", "RegDate"], inplace=True)
    return df.groupby("Code", as_index=False).last()


def plot_chart(sub_df, pivots, name, save_path=None):
    sub_df = sub_df.tail(90).copy()
    sub_df.reset_index(drop=True, inplace=True)

    sub_df["Date_dt"] = pd.to_datetime(sub_df["Date"])
    sub_df["XIndex"]  = np.arange(len(sub_df))

    intervals = find_contraction_intervals(sub_df, peak_lookback=90, min_bars_required=30)

    ohlc_vals = sub_df[["XIndex","Open","High","Low","Close"]].values
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True,
                                   gridspec_kw={"height_ratios":[3,1]},
                                   figsize=(10,8))
    candlestick_ohlc(ax1, ohlc_vals, width=0.6, colorup="r", colordown="b")

    pvt = pivots.get("Pivot", np.nan)
    btm = pivots.get("Bottom", np.nan)
    st_date_str = pivots.get("LastIntStart","")
    ed_date_str = pivots.get("LastIntEnd","")

    def date_to_xindex(d_str):
        try:
            dt_ = pd.to_datetime(d_str)
        except:
            return None
        matched = sub_df[sub_df["Date_dt"]== dt_]
        if not matched.empty:
            return matched.iloc[0]["XIndex"]
        return None

    x_s = date_to_xindex(st_date_str)
    x_e = date_to_xindex(ed_date_str)

    if (x_s is not None) and (x_e is not None) and (x_s <= x_e):
        if pd.notna(btm):
            ax1.hlines(y=btm, xmin=x_s, xmax=x_e, color="black", linestyles="-", linewidth=1, alpha=0.8)
        if pd.notna(pvt):
            ax1.hlines(y=pvt, xmin=x_s, xmax=x_e, color="black", linestyles="-", linewidth=1, alpha=0.8)

    if x_e is not None:
        if (x_e>=0) and (x_e< len(sub_df)):
            ax1.axvline(x=x_e, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    if pd.notna(pvt) and pvt>0:
        ax1.axhline(y=pvt, color="magenta", linestyle="--", linewidth=0.5, label="Pivot")
        ax1.text(0.5, pvt, f"{pvt:,}", fontsize=8, color="magenta",
                 verticalalignment='bottom', horizontalalignment='left')

    if pd.notna(btm) and btm>0:
        ax1.axhline(y=btm, color="gray", linestyle="--", linewidth=0.5, label="Bottom")
        ax1.text(0.5, btm, f"{btm:,}", fontsize=8, color="gray",
                 verticalalignment='bottom', horizontalalignment='left')

    for sma, c in zip(["SMA_5","SMA_10","SMA_20","SMA_50"],
                      ["red","orange","lightgreen","blue"]):
        if sma in sub_df.columns:
            ax1.plot(sub_df["XIndex"], sub_df[sma], color=c, linewidth=0.5)

    if "BBAND_U_200" in sub_df.columns:
        ax1.plot(sub_df["XIndex"], sub_df["BBAND_U_200"], color="orange", linewidth=1)

    sub_df["VolColor"] = np.where(sub_df["Close"]>= sub_df["Open"], "r","b")
    ax2.bar(sub_df["XIndex"], sub_df["Volume"], width=0.6, color=sub_df["VolColor"])

    ax2.set_xticks(sub_df["XIndex"][::10])
    ax2.set_xticklabels(sub_df["Date_dt"].dt.strftime("%Y-%m-%d")[::10], rotation=45, fontsize=8)
    for lb in ax1.get_yticklabels():
        lb.set_fontsize(8)
    for lb in ax2.get_yticklabels():
        lb.set_fontsize(8)

    ax1.set_title(f"{name}", fontsize=11)

    # 수축 구간 표시
    for c_int in intervals:
        i_s = c_int["start"]
        i_e = c_int["end"]
        mn  = c_int["min_price"]
        mx  = c_int["max_price"]
        if i_s<0: i_s=0
        if i_e>(len(sub_df)-1): i_e= len(sub_df)-1
        ax1.hlines(y=mn, xmin=i_s, xmax=i_e, color="black", linestyle="--", linewidth=0.5, alpha=0.6)
        ax1.hlines(y=mx, xmin=i_s, xmax=i_e, color="black", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    return fig


# ------------------------------
# 메인 UI (기존 코드 대부분 유지)
# ------------------------------
st.set_page_config(page_title="MVCP 후보 관리", layout="wide")
st.markdown("<br><h3>MVCP 후보 관리</h3>", unsafe_allow_html=True)

if not os.path.exists(CSV_FILE) or not os.path.exists(PARQUET_FILE):
    st.error("CSV 파일 또는 KRX 데이터 파일이 없습니다.")
    st.stop()

cands_df = load_candidate_list(CSV_FILE)
cands_df["RegDate"] = pd.to_datetime(cands_df["RegDate"])
cands_df = cands_df.sort_values("RegDate", ascending=False).reset_index(drop=True)

df_krx = load_krx_data(PARQUET_FILE)

st.markdown("<BR>", unsafe_allow_html=True)
st.markdown(f"총 {len(cands_df)} 종목 관리 중")
st.markdown("<BR>", unsafe_allow_html=True)

show_only_breakouts = st.checkbox("피벗 돌파 후 종목만 보기", value=False)
show_only_breakdowns = st.checkbox("VCP 이탈 종목만 보기", value=False)
st.markdown("<BR>", unsafe_allow_html=True)

df_merged = cands_df.copy()
df_merged["Bottom"] = pd.to_numeric(df_merged["Bottom"], errors="coerce")
df_merged["Pivot"]  = pd.to_numeric(df_merged["Pivot"], errors="coerce")
df_merged["T"]      = pd.to_numeric(df_merged["T"], errors="coerce")

# 최근 종가 매핑
latest_close_map = {}
for code in df_merged["Code"].unique():
    sub_ = df_krx[df_krx["Code"]== code]
    if not sub_.empty:
        latest_close_map[code] = sub_.iloc[-1]["Close"]
    else:
        latest_close_map[code] = np.nan

df_merged["today_close"] = df_merged["Code"].map(latest_close_map)
df_merged["is_breakdown"] = df_merged.apply(
    lambda row: (pd.notna(row["Bottom"]) and pd.notna(row["today_close"]) and row["today_close"]< row["Bottom"]),
    axis=1
)

selected = cands_df["Name"].tolist()

for _, row in df_merged.iterrows():
    name_ = row["Name"]
    code_ = row["Code"]
    if name_ not in selected:
        continue

    sub = df_krx[df_krx["Code"]== code_].copy()
    if sub.empty:
        continue

    latest_close= row["today_close"]
    pivot= row["Pivot"]
    btm= row["Bottom"]
    t_count= row.get("T", np.nan)

    last_vol_pct = None
    if pd.notna(pivot) and pivot>0 and pd.notna(btm) and btm>0:
        last_vol_pct = (pivot - btm)/ btm *100

    # 돌파 여부
    is_breakout= False
    breakout_pct = 0.0
    if pd.notna(pivot) and (latest_close> pivot):
        is_breakout= True
        # ADD: 피벗 대비 상승률
        if pivot>0:
            breakout_pct = (latest_close - pivot) / pivot *100

    # 이탈 여부
    is_breakdown= row["is_breakdown"]

    # 필터
    if show_only_breakouts and (not is_breakout):
        continue
    if show_only_breakdowns and (not is_breakdown):
        continue

    # 상태 문구
    status_info=[]
    if is_breakout:
        # 기존: status_info.append(" [ 돌파 후 ]")
        # 수정: 상승률 추가
        status_info.append(f" [ 돌파 후 {breakout_pct:.1f}% ]")
    if is_breakdown:
        status_info.append(" [ 이탈 ]")
    stitle = " / ".join(status_info) if status_info else ""

    # ---------------------------
    # 섹션 제목 만들기
    # ---------------------------
    pivot_str = f"{pivot:,.0f}" if pd.notna(pivot) else "-"
    btm_str   = f"{btm:,.0f}" if pd.notna(btm) else "-"
    vol_str   = f"{last_vol_pct:.1f}%" if last_vol_pct is not None else "-"
    t_str     = f"{int(t_count)}" if pd.notna(t_count) else "-"
    title_exp = (f"{name_} ({code_})&nbsp;&nbsp;C {latest_close:,.0f} "
                 f"&nbsp;&nbsp;V {pivot_str}&nbsp;&nbsp;B {btm_str}&nbsp;&nbsp;T {t_str}&nbsp;&nbsp;Last T {vol_str}&nbsp;&nbsp;{stitle}")

    with st.expander(title_exp):
        fig= plot_chart(sub, row, name_)
        st.pyplot(fig)
        buf= io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)

        col1, col2, col3= st.columns([2,2,2])
        with col2:
            # 개별 삭제
            with st.form(f"delete_form_{code_}"):
                pw= st.text_input("비밀번호", type="password", key=f"pw_{code_}", label_visibility="collapsed")
                submitted= st.form_submit_button("종목 삭제")
                if submitted:
                    if pw== PASSWORD:
                        orig= pd.read_csv(CSV_FILE, dtype={"Code":str})
                        updated= orig[orig["Code"]!= code_].copy()
                        updated.to_csv(CSV_FILE, index=False)
                        st.success(f"{name_} ({code_}) 삭제 완료")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("비밀번호가 일치하지 않습니다.")
        with col3:
            with st.form(f"blacklist_{code_}"):
                pw= st.text_input("비밀번호", type="password", key=f"blkpw_{code_}", label_visibility="collapsed")
                submitted= st.form_submit_button("블랙리스트 추가")
                if submitted:
                    if pw== PASSWORD:
                        blacklist_file = "blacklist.csv"
                        if os.path.exists(blacklist_file):
                            df_bl= pd.read_csv(blacklist_file, dtype={"Code":str})
                        else:
                            df_bl= pd.DataFrame(columns=["Code"])
            
                        new_row= pd.DataFrame({"Code":[code_]})
                        df_merged= pd.concat([df_bl, new_row], ignore_index=True)
                        df_merged.drop_duplicates(["Code"], keep="last", inplace=True)
                        df_merged.to_csv(blacklist_file, index=False)
                        
                        orig= pd.read_csv(CSV_FILE, dtype={"Code":str})
                        updated= orig[orig["Code"]!= code_].copy()
                        updated.to_csv(CSV_FILE, index=False)
                        st.info(f"블랙리스트 등록 완료: {code_}")

                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("비밀번호가 일치하지 않습니다.")
        plt.close(fig)

codes_to_remove = df_merged.loc[df_merged["is_breakdown"], "Code"].unique().tolist()
col21, col22 = st.columns([3,1])
with col22:
    with st.form("delete_all_breakdowns"):
        st.markdown(f"이탈 종목 일괄 삭제 : {len(codes_to_remove)} 종목")
        pw_all = st.text_input("권한 비밀번호", type="password", label_visibility="collapsed")
        submitted_all = st.form_submit_button("일괄 삭제")
        if submitted_all:
            if pw_all== PASSWORD:
                if not codes_to_remove:
                    st.info("이탈 종목이 없습니다.")
                else:
                    orig= pd.read_csv(CSV_FILE, dtype={"Code":str})
                    updated= orig[~orig["Code"].isin(codes_to_remove)].copy()
                    updated.to_csv(CSV_FILE, index=False)
                    st.success(f"이탈 종목 {len(codes_to_remove)} 종목 일괄 삭제 완료")
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.warning("비밀번호가 일치하지 않습니다.")

coll, colr= st.columns([9,1])
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