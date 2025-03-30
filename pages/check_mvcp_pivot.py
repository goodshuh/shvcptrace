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

PASSWORD = "236135"

def load_krxdata(parquet_path):
    df = pd.read_parquet(parquet_path)
    df.dropna(subset=["Open","High","Low","Close","Volume"], inplace=True)
    return df[df["Volume"]>0].copy()

def find_90day_peak_info(gdf):
    """ 기존 find_60day_peak_info -> 90일로 변경 """
    lookback = min(len(gdf), 90)  # CHANGED 60 -> 90
    sub = gdf.iloc[-lookback:]
    peak_value = sub["Close"].max()
    peak_idx  = sub["Close"].idxmax()
    return peak_value, peak_idx

def check_base_2weeks_after_peak(gdf, peak_idx):
    positions = list(gdf.index)
    if peak_idx not in positions:
        return False
    peak_pos = positions.index(peak_idx)
    return (len(positions) - 1 - peak_pos) >= 9

def check_40pct_no_break(gdf, peak_value, maxdrop_limit):
    return not (gdf["Low"] < (1 - maxdrop_limit)* peak_value).any()

def find_contraction_intervals(df, peak_lookback=90, min_bars_required=90):
    """
    글로벌 수축구간 탐색 (전 인덱스 기준)
    -> 반환 intervals: [{'start': idxA, 'end': idxB, 'min_price':..., 'max_price':...}, ...]
    """
    if len(df) < min_bars_required:
        return []
    end_idx = df.index[-1]
    start_idx = end_idx - peak_lookback + 1
    if start_idx < df.index[0]:
        start_idx = df.index[0]

    sub = df.loc[start_idx:end_idx]
    if sub.empty:
        return []

    peak_local_idx = sub["Close"].idxmax()
    if peak_local_idx == sub.index[-1]:
        return []

    cur_start = peak_local_idx + 1
    if cur_start > sub.index[-1]:
        return []

    intervals = []
    def calc_vol(itv):
        lo= itv["min_price"]
        hi= itv["max_price"]
        return (hi/lo -1.0) if lo>0 else 0

    def finalize_and_merge():
        while len(intervals)>=2:
            l_ = intervals[-1]
            p_ = intervals[-2]
            v_l= calc_vol(l_)
            v_p= calc_vol(p_)
            cond1= (v_l> v_p)
            cond2= (l_["min_price"]< p_["min_price"])
            if cond1 or cond2:
                merged= {
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

    # 첫 interval
    if cur_start not in df.index:
        return []
    intervals.append({
        "start": cur_start,
        "end":   cur_start,
        "min_price": float(df.loc[cur_start,"Low"]),
        "max_price": float(df.loc[cur_start,"High"])
    })

    idxs= sub.index[sub.index> cur_start]
    for i in idxs:
        day_low   = float(df.loc[i,"Low"])
        day_high  = float(df.loc[i,"High"])
        day_close = float(df.loc[i,"Close"])

        c_int= intervals[-1]
        c_int["end"]= i

        cur_vol= calc_vol(c_int)

        # fake down check
        if day_low< c_int["min_price"]:
            old_min= c_int["min_price"]
            if cur_vol<= 0.065:
                if day_close< old_min:
                    c_int["min_price"]= day_low
                else:
                    pass
            else:
                c_int["min_price"]= day_low

        if day_high> c_int["max_price"]:
            if day_close<= c_int["max_price"]:
                c_int["max_price"]= day_high
            else:
                finalize_and_merge()
                intervals.append({
                    "start": i,
                    "end":   i,
                    "min_price": day_low,
                    "max_price": day_high
                })

        # 최근 3일 변동폭
        if (i- c_int["start"])>=2:
            vol_c= calc_vol(c_int)
            sub_idx= df.index[(df.index>= c_int["start"]) & (df.index<= i)]
            if len(sub_idx)>=3:
                last3= sub_idx[-3:]
                sl_low= df.loc[last3,"Low"].min()
                sl_hi = df.loc[last3,"High"].max()
                rec_vol= (sl_hi/sl_low-1.0) if sl_low>0 else 0
                if rec_vol<= vol_c*0.7:
                    finalize_and_merge()
                    intervals.append({
                        "start": i,
                        "end":   i,
                        "min_price": day_low,
                        "max_price": day_high
                    })

    finalize_and_merge()
    return intervals

def is_priority_by_bigvol_close(gdf, last_close, lookback=50):
    sub= gdf.tail(lookback)
    if len(sub)<1:
        return False
    sub_s= sub.sort_values("Volume", ascending=False)
    t1= sub_s.iloc[0]["Close"]
    t2= sub_s.iloc[1]["Close"] if len(sub_s)>1 else None
    def w2pct(x):
        return 0<((last_close- x)/ x)<=0.02
    if w2pct(t1): return True
    if t2 and w2pct(t2): return True
    return False

def screen_candidates(gdf_, day_offset, maxdrop_limit=0.4, min_contractions=3):
    """
    기준일 = idx_basis
    기준일 직전까지를 수축 구간(subdf)으로 검사
    """

    res_list = []
    # 데이터가 너무 짧으면 불가 (예시 90일 필요)
    if len(gdf_) < 90:
        return res_list

    # idx_basis = '기준일' 인덱스
    idx_basis = len(gdf_) - 1 - day_offset
    # 만약 idx_basis <= 0이면 subdf가 비거나 너무 짧아짐
    if idx_basis <= 0:
        return res_list

    # 수축 구간: ~ idx_basis-1
    subdf = gdf_.iloc[:idx_basis].copy()  # 기준일 전까지
    if len(subdf) < 90:
        return res_list

    # basis_candle = gdf_.iloc[idx_basis] (오프셋 문제 수정)
    if idx_basis >= len(gdf_):
        return res_list
    basis_candle = gdf_.iloc[idx_basis]

    # (A) 90일 peak
    peak_val, peak_idx = find_90day_peak_info(subdf)
    if not peak_val or peak_idx is None:
        return res_list

    # 2주 경과
    if not check_base_2weeks_after_peak(subdf, peak_idx):
        return res_list

    sub_after = subdf.loc[peak_idx:]
    if not check_40pct_no_break(sub_after, peak_val, maxdrop_limit):
        return res_list

    basis_close = basis_candle["Close"]
    if basis_close <= 0:
        return res_list

    drop_from_peak = (peak_val - basis_close) / peak_val
    if drop_from_peak > 0.27:
        return res_list

    # SMA
    lastrow = subdf.iloc[-1]
    sma50 = lastrow.get("SMA_50")
    sma100= lastrow.get("SMA_100")
    sma200= lastrow.get("SMA_200")
    if not (sma50 and sma100 and sma200):
        return res_list
    if not (sma50 > sma100 > sma200):
        return res_list
    if lastrow["Close"] <= sma50:
        return res_list

    # 수축 구간
    intervals = find_contraction_intervals(subdf, peak_lookback=90, min_bars_required=90)
    if len(intervals) < min_contractions:
        return res_list

    t_count = len(intervals)
    last_int = intervals[-1]
    hi_ = subdf.loc[last_int["start"] : last_int["end"], "High"].max()
    lo_ = subdf.loc[last_int["start"] : last_int["end"], "Low"].min()
    if lo_ <= 0:
        return res_list
    vol_ratio = (hi_/lo_ - 1.0)
    if vol_ratio > 0.065:
        return res_list

    if len(intervals) >= 2:
        prev_int= intervals[-2]
        vol_last= subdf.loc[last_int["start"] : last_int["end"], "Volume"].mean()
        vol_prev= subdf.loc[prev_int["start"] : prev_int["end"], "Volume"].mean()
        if vol_last >= vol_prev * 1.5: #이전 구간 거래량보다 50% 이상 증가시 탈락
            return res_list

    nm = lastrow["Name"]
    cd = lastrow["Code"]
    dt_ = lastrow["Date"]

    # 우선순위
    prio = is_priority_by_bigvol_close(subdf, lastrow["Close"])

    pivot_price  = hi_
    bottom_price = lo_
    st_idx= last_int["start"]
    ed_idx= last_int["end"]
    last_period= ed_idx - st_idx + 1
    st_date= subdf.loc[st_idx,"Date"]
    ed_date= subdf.loc[ed_idx,"Date"]

    # ---------------------
    # 기준일 캔들
    # ---------------------
    bc_close= basis_candle["Close"]
    bc_open = basis_candle["Open"]
    bc_high = basis_candle["High"]
    bc_low  = basis_candle["Low"]
    bc_vol  = basis_candle["Volume"]
    bc_date= basis_candle["Date"]

    breakout= "N"
    breakout_date= ""
    new_interval= "N"

    condAbove= (bc_close > hi_)

    # prev_top
    prev_top= 0
    if len(intervals)>=2:
        hi2 = subdf.loc[intervals[-2]["start"]: intervals[-2]["end"], "High"].max()
        prev_top= hi2
    condPT= True
    if prev_top>0:
        condPT= (bc_close> prev_top)

    # 전일 거래량
    condVol2= False
    if (idx_basis-1) >= 0:
        prev_vol= gdf_.iloc[idx_basis-1]["Volume"]
        condVol2= (bc_vol >= 2* prev_vol)
    else:
        prev_vol= np.nan

    # 최근 50일 평균
    tail50= gdf_.iloc[:idx_basis].tail(50)
    vol50= tail50["Volume"].mean() if len(tail50)>0 else 0
    condVol50= (bc_vol>= vol50)

    ext_hi= max(hi_, bc_high)
    ext_low= min(lo_, bc_low)
    ext_vol= (ext_hi / ext_low -1.0) if ext_low>0 else 0

    condVolUp= False
    if (idx_basis-1) >= 0 and not np.isnan(prev_vol):
        condVolUp= (bc_vol > prev_vol)

    if condAbove:
        if condPT and condVol2 and condVol50:
            breakout= "Y"
            breakout_date= bc_date
        else:
            if condVolUp:
                # 신규 구간
                new_interval= "Y"
                pivot_price= bc_high
                bottom_price= bc_low
                st_date= bc_date
                ed_date= bc_date
                last_period=1
                new_vol= (bc_high/bc_low - 1.0) if bc_low>0 else 0
                if new_vol>0.065:
                    return res_list
            else:
                # 연장
                if ext_vol>0.065:
                    return res_list
                else:
                    if bc_low < lo_ and bc_close < lo_:
                        bottom_price = bc_low
                    else:
                        bottom_price = lo_

                    ed_date = bc_date  # 구간 종료일 업데이트
                    last_period = (bc_date - st_date).days + 1 if isinstance(bc_date, pd.Timestamp) and isinstance(st_date, pd.Timestamp) else last_period
    else:
        # 연장
        if ext_vol>0.065:
            return res_list
        else:
            if bc_low < lo_ and bc_close < lo_:
                bottom_price = bc_low
            else:
                bottom_price = lo_

            ed_date = bc_date  # 구간 종료일 업데이트
            last_period = (bc_date - st_date).days + 1 if isinstance(bc_date, pd.Timestamp) and isinstance(st_date, pd.Timestamp) else last_period

    final_dt   = bc_date
    final_close= bc_close

    res_list.append((
        cd, nm, final_dt, final_close,
        pivot_price, bottom_price,
        st_date, ed_date,
        last_period, len(intervals),
        prio,
        breakout,
        breakout_date,
        new_interval
    ))
    return res_list

# ===========================================
# 예시: 차트 표시
# ===========================================
def plot_chart_with_intervals(df, intervals, lookback=90):
    """
    하나의 df 전체(글로벌 인덱스),
    intervals: find_contraction_intervals(df)의 결과
    -> 차트는 최근 `lookback` 일만
    -> intervals 중 겹치는 부분만 hlines 표시
    """
    sub_chart = df.iloc[-lookback:].copy()
    sub_chart.reset_index(drop=True, inplace=True)
    sub_chart["XIndex"] = np.arange(len(sub_chart))

    # 실제 전역 인덱스에서
    chart_idx_min = df.index[-lookback]  # 첫행
    chart_idx_max = df.index[-1]         # 마지막행

    # candlestick data
    ohlc = sub_chart[["XIndex","Open","High","Low","Close"]].values

    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize=(10,8),
                                   gridspec_kw={"height_ratios":[3,1]})
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup="r", colordown="b")

    # intervals 표시
    for c_int in intervals:
        i_s = c_int["start"]
        i_e = c_int["end"]
        mn  = c_int["min_price"]
        mx  = c_int["max_price"]

        # 구간이 차트 범위와 겹치는지?
        if (i_e < chart_idx_min) or (i_s > chart_idx_max):
            continue  # 전혀 안겹침 -> skip

        # 범위 클램핑
        clamp_s = max(i_s, chart_idx_min)
        clamp_e = min(i_e, chart_idx_max)

        # 로컬 x
        local_s = clamp_s - chart_idx_min
        local_e = clamp_e - chart_idx_min

        # hlines
        ax1.hlines(y=mn, xmin=local_s, xmax=local_e, color="black", linestyle="--", linewidth=0.5, alpha=0.6)
        ax1.hlines(y=mx, xmin=local_s, xmax=local_e, color="black", linestyle="--", linewidth=0.5, alpha=0.6)

    # 거래량
    sub_chart["VolColor"] = np.where(sub_chart["Close"]>= sub_chart["Open"], "r","b")
    ax2.bar(sub_chart["XIndex"], sub_chart["Volume"], width=0.6, color=sub_chart["VolColor"])

    # x ticks
    xvals = sub_chart["XIndex"].values
    dt_labels = sub_chart["Date"].dt.strftime("%Y-%m-%d").values
    ax2.set_xticks(xvals[::10])
    ax2.set_xticklabels(dt_labels[::10], rotation=45)

    return fig

# ===========================================
# Streamlit main
# ===========================================
st.set_page_config(page_title="MVCP 피벗 후보 스크리닝", layout="wide")
st.markdown("<br><h3>MVCP 피벗 후보 스크리닝</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

path= "Z. krxdata.parquet"
save_folder= "Z. VCP PV-"
blacklist_file = "blacklist.csv"

col1, col2, col3= st.columns(3)
with col1:
    day_offset= st.number_input("기준일 설정 (n 일 전)", 0,100,0)
    mcap_threshold= st.slider("시가총액 (억, 이상)",0,10000,3000,step=500)
    maxdrop_limit= st.slider("전고 이후 최대 하락률 (%, 이하)",0,100,40)
    min_contr= st.slider("변동성 축소 횟수 (최소)",1,5,3)
    show_bbu200= st.checkbox("BB200 라인 표시", True)
    save_chart= True

    save_list_file= st.checkbox("리스트 서버 저장", value=False)
    user_password= ""
    if save_list_file:
        user_password= st.text_input("권한 비밀번호", type="password")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("스크리닝 실행"):
    st.markdown("<br>", unsafe_allow_html=True)
    df= load_krxdata(path)
    df.sort_values(["Code","Date"], inplace=True)

    ref_idx= df.index[-1]- day_offset if len(df)>day_offset else 0
    if ref_idx in df.index:
        refdate= df.loc[ref_idx,"Date"].strftime("%Y-%m-%d")
    else:
        refdate= "noData"

    grouped= df.groupby("Code", group_keys=False)
    all_candidates= []

    for code, gdf in grouped:
        gdf= gdf.reset_index(drop=True)
        if gdf.iloc[-1]["MarketCap"]< mcap_threshold*1e8:
            continue
        
        cands= screen_candidates(
            gdf,
            day_offset= day_offset,
            maxdrop_limit= maxdrop_limit/100,
            min_contractions= min_contr
        )
        if cands:
            all_candidates.extend(cands)


    # ============= BLACKLIST STEP ==============
    # 1) blacklist.csv 읽고
    black_codes= set()
    if os.path.exists(blacklist_file):
        df_bl = pd.read_csv(blacklist_file, dtype={"Code":str})
        black_codes= set(df_bl["Code"].unique())

    # 2) all_candidates에서 black_codes에 해당되는 종목 제거
    #    cand 구조: (cd, nm, ...)
    tmp_list= []
    for cand in all_candidates:
        cd= cand[0]  # code
        if cd not in black_codes:
            tmp_list.append(cand)
    all_candidates= tmp_list
    # ===========================================

    st.subheader(f"{refdate}\n스크리닝 결과 : {len(all_candidates)} 종목")

    save_csv= "Z. list_mvcppivot.csv"
    if all_candidates:
        if save_list_file:
            if user_password== PASSWORD:
                recs=[]
                for cand in all_candidates:
                    # (cd, nm, fdt, fcls, pvt, btm, st_, ed_, lp, t_, pr, brk, brkdt, newi)
                    (cd, nm, fdt, fcls, pvt, btm, st_, ed_, lp, t_, pr, brk, brkdt, newi)= cand
                    
                    brk_bool= "True" if brk=="Y" else "False"
                    new_bool= "True" if newi=="Y" else "False"

                    recs.append({
                        "Name": nm,
                        "Code": cd,
                        "RegDate": fdt.strftime("%Y-%m-%d") if isinstance(fdt,pd.Timestamp) else str(fdt),
                        "LastClose": fcls,
                        "Pivot": pvt,
                        "Bottom": btm,
                        "LastIntStart": st_.strftime("%Y-%m-%d") if isinstance(st_,pd.Timestamp) else str(st_),
                        "LastIntEnd": ed_.strftime("%Y-%m-%d") if isinstance(ed_,pd.Timestamp) else str(ed_),
                        "LastIntPeriod": lp,
                        "T": t_,
                        "Breakout": brk_bool,
                        "BreakoutDate": pd.to_datetime(brkdt).strftime("%Y-%m-%d") if (brk=="Y" and brkdt) else "",
                        "NewInterval": new_bool
                    })
                
                newdf= pd.DataFrame(recs, columns=[
                    "Name","Code","RegDate","LastClose","Pivot","Bottom",
                    "LastIntStart","LastIntEnd","LastIntPeriod","T",
                    "Breakout","BreakoutDate","NewInterval"
                ])
                newdf["Code"]= newdf["Code"].astype(str).str.zfill(6)

                if os.path.exists(save_csv):
                    olddf= pd.read_csv(save_csv, dtype={"Code":str})
                    merged= pd.concat([olddf,newdf], ignore_index=True)
                    merged.drop_duplicates(subset=["Name","RegDate"], keep="last", inplace=True)
                    merged.sort_values(["Name","RegDate"], inplace=True)
                    merged.to_csv(save_csv, index=False)
                    st.info("리스트 갱신 완료 (PW OK)")
                else:
                    newdf.sort_values(["Name","RegDate"], inplace=True)
                    newdf.to_csv(save_csv, index=False)
                    st.info("리스트 생성 완료 (PW OK)")
            else:
                st.warning("비밀번호가 일치하지 않습니다. 파일 저장이 진행되지 않습니다.")

    if save_chart:
        os.makedirs(save_folder, exist_ok=True)

    # 차트
    for cand in all_candidates:
        (cd, nm, fdt, fcls, pvt, btm, st_, ed_, lp, t_, pr, brk, brkdt, newi)= cand
        
        sub= df[df["Code"]== cd].reset_index(drop=True)
        idx_= len(sub)-1- day_offset
        if idx_<0:
            continue

        # CHANGED from 60 -> 90
        sub_chart= sub.iloc[: idx_+1].tail(90).copy()
        sub_chart.reset_index(drop=True, inplace=True)
        sub_chart["XIndex"]= np.arange(len(sub_chart))

        ohlc= sub_chart[["XIndex","Open","High","Low","Close"]].values
        fig, (ax1, ax2)= plt.subplots(2,1,sharex=True,
                                      gridspec_kw={"height_ratios":[3,1]},
                                      figsize=(10,8))
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup="r", colordown="b")

        for sma, color in zip(["SMA_5","SMA_10","SMA_20","SMA_50"],
                              ["red","orange","lightgreen","blue"]):
            if sma in sub_chart.columns:
                ax1.plot(sub_chart["XIndex"], sub_chart[sma], color=color, linewidth=0.5)

        if show_bbu200 and ("BBAND_U_200" in sub_chart.columns):
            ax1.plot(sub_chart["XIndex"], sub_chart["BBAND_U_200"], color="orange", linewidth=0.8)

        sub_chart["VolColor"]= np.where(sub_chart["Close"]>= sub_chart["Open"], "r","b")
        ax2.bar(sub_chart["XIndex"], sub_chart["Volume"], width=0.6, color=sub_chart["VolColor"])

        xind= sub_chart["XIndex"].values
        dt_labels= pd.to_datetime(sub_chart["Date"]).dt.strftime("%Y-%m-%d").values
        ticks= xind[::10]
        labs= dt_labels[::10]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labs, rotation=45, fontsize=8)

        for lb in ax1.get_yticklabels():
            lb.set_fontsize(8)
        for lb in ax2.get_yticklabels():
            lb.set_fontsize(8)

        if pvt>0:
            ax1.axhline(y=pvt, color="magenta", linestyle="--", linewidth=0.5)
            ax1.text(sub_chart["XIndex"].min() + 0.5, pvt, f"{pvt:,}", fontsize=8, color="magenta",
                     verticalalignment='bottom', horizontalalignment='left')

        # local intervals: 90
        def find_contraction_intervals_local(df, peak_lookback=90, min_bars_required=90):
            """
            차트 표시용 동일 수정
            """
            if len(df)< min_bars_required:
                return []
            end_i= len(df)-1
            st_i= max(0, end_i- peak_lookback+1)
            s= df.iloc[st_i:end_i+1].copy().reset_index(drop=True)
            if s.empty:
                return []

            pk= s["Close"].idxmax()
            if pk>= len(s)-1:
                return []
            cs= pk+1
            if cs>= len(s):
                return []
            
            intervals2=[]
            def calcv(ii):
                lo= ii["min_price"]
                hi= ii["max_price"]
                return (hi/lo -1.0) if lo>0 else 0
            def fm2():
                while len(intervals2)>=2:
                    l_ = intervals2[-1]
                    p_ = intervals2[-2]
                    vl= calcv(l_)
                    vp= calcv(p_)
                    c1= (vl>vp)
                    c2= (l_["min_price"]< p_["min_price"])
                    if c1 or c2:
                        m2= {
                            "start": p_["start"],
                            "end":   l_["end"],
                            "min_price": min(p_["min_price"], l_["min_price"]),
                            "max_price": max(p_["max_price"], l_["max_price"])
                        }
                        intervals2.pop()
                        intervals2.pop()
                        intervals2.append(m2)
                    else:
                        break

            intervals2.append({
                "start": cs,
                "end": cs,
                "min_price": float(s.loc[cs,"Low"]),
                "max_price": float(s.loc[cs,"High"])
            })

            for i in range(cs+1, len(s)):
                lw= float(s.loc[i,"Low"])
                hg= float(s.loc[i,"High"])
                cl= float(s.loc[i,"Close"])

                c_= intervals2[-1]
                c_["end"]= i
                cur_vol= calcv(c_)

                if lw< c_["min_price"]:
                    old_min= c_["min_price"]
                    if cur_vol<=0.065:
                        if cl< old_min:
                            c_["min_price"]= lw
                        else:
                            pass
                    else:
                        c_["min_price"]= lw

                if hg> c_["max_price"]:
                    if cl<= c_["max_price"]:
                        c_["max_price"]= hg
                    else:
                        fm2()
                        intervals2.append({
                            "start": i,
                            "end": i,
                            "min_price": lw,
                            "max_price": hg
                        })
                if (i- c_["start"])>=2:
                    vc_= calcv(c_)
                    rng_= range(c_["start"], i+1)
                    if len(rng_)>=3:
                        last3_= list(rng_)[-3:]
                        lows_= [s.loc[x,"Low"] for x in last3_]
                        his_ = [s.loc[x,"High"] for x in last3_]
                        sl_low= min(lows_)
                        sl_hi= max(his_)
                        rv_= (sl_hi/sl_low -1.0) if sl_low>0 else 0
                        if rv_<= vc_*0.7:
                            fm2()
                            intervals2.append({
                                "start": i,
                                "end": i,
                                "min_price": lw,
                                "max_price": hg
                            })
            fm2()
            return intervals2

        intervals_local= find_contraction_intervals_local(sub_chart, 90, 90)
        for ci in intervals_local:
            i_s= ci["start"]
            i_e= ci["end"]
            if i_s<0: i_s=0
            if i_e> len(sub_chart)-1: i_e= len(sub_chart)-1
            mn= ci["min_price"]
            mx= ci["max_price"]
            ax1.hlines(y=mn, xmin=i_s, xmax=i_e, color="black", linestyles="--", linewidth=0.5, alpha=0.6)
            ax1.hlines(y=mx, xmin=i_s, xmax=i_e, color="black", linestyles="--", linewidth=0.5, alpha=0.6)

        # 돌파/신규 문구
        brk_str = ("돌파" if brk=="Y" else "")
        newi_str= ("신규 상승 T" if newi=="Y" else "")

        dt_str = pd.to_datetime(fdt).strftime("%Y-%m-%d") if isinstance(fdt,pd.Timestamp) else str(fdt)
        extra_str_list=[]
        if brk_str:
            extra_str_list.append(brk_str)
        if newi_str:
            extra_str_list.append(newi_str)
        extra_str = " | ".join(extra_str_list) if extra_str_list else ""

        title= f"{dt_str} {nm}{'**' if pr else ''} ({cd})  종가 {fcls:,}  피벗 {pvt:,}  T {t_}{'  '+extra_str if extra_str else ''}"
        ax1.set_title(title, fontsize=10)
        title= f"{nm}{'**' if pr else ''} ({cd}) C {fcls:,} V {pvt:,} T {t_}{' &nbsp;&nbsp; '+extra_str if extra_str else ''}"
        
        with st.expander(title):
            st.pyplot(fig)
            buf= io.BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            fname_dt= pd.to_datetime(fdt).strftime("%Y-%m-%d") if isinstance(fdt,pd.Timestamp) else str(fdt)
            fname= f"{fname_dt}_{nm}".replace(" ","_").replace("/","_") + ".png"

            #col_a, col_b = st.columns([2,2])
            #with col_a:
            #    st.download_button(
            #        label="차트 다운로드",
            #        data=buf.getvalue(),
            #        file_name=fname,
            #        mime="image/png"
            #    )
            #with col_b:
            # 
            #   if st.button("블랙리스트 추가", key=f"blk_{cd}"):
            #            # 1) blacklist.csv 읽기
            #            if os.path.exists(blacklist_file):
            #                df_bl= pd.read_csv(blacklist_file, dtype={"Code":str})
            #            else:
            #                df_bl= pd.DataFrame(columns=["Code"])
            #
            #            # 2) 추가
            #            new_row= pd.DataFrame({"Code":[cd]})
            #            df_merged= pd.concat([df_bl, new_row], ignore_index=True)
            #            df_merged.drop_duplicates(["Code"], keep="last", inplace=True)
            #            df_merged.to_csv(blacklist_file, index=False)
            #            st.info(f"블랙리스트 등록 완료: {cd}")
            #
            #            st.experimental_rerun()

            if save_chart:
                with open(os.path.join(save_folder, fname), "wb") as ff:
                    ff.write(buf.getvalue())
            
        plt.close(fig)

# BACK
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