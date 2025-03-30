# Streamlit 런처 앱 - 각 파일별 실행 버튼 UI (최종 정리본)
import streamlit as st
import subprocess
import os

st.set_page_config(page_title="SH VCP MANAGEMENT", layout="wide")
st.markdown("<br><h3>SH VCP MANAGEMENT</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("아래 파란색 메뉴 버튼을 눌러 원하는 스크리닝 방식을 선택하세요.", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([3, 1, 3, 1, 3])

with col1:
    st.info("메인 기능")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/check_mvcp_pivot" target="_self">
        <div style="
            display: block;
            background-color: #007AFF;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">MVCP 스크리닝</div>
    </a>
    """,
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/manage_mvcp_list" target="_self">
        <div style="
            display: block;
            background-color: #007AFF;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">MVCP 후보 관리</div>
    </a>
    """,
    unsafe_allow_html=True)

with col3:
    st.info("서브 기능")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/check_bb200" target="_self">
        <div style="
            display: block;
            background-color: #007AFF;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">BB200 돌파 스크리닝</div>
    </a>
    """,
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/check_vol_pivot" target="_self">
        <div style="
            display: block;
            background-color: #007AFF;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">볼륨 피벗 돌파 스크리닝</div>
    </a>
    """,
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/check_vcp_pivot" target="_self">
        <div style="
            display: block;
            background-color: #007AFF;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">VCP 스크리닝</div>
    </a>
    """,
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/manage_vcp_list" target="_self">
        <div style="
            display: block;
            background-color: #007AFF;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">VCP 후보 관리</div>
    </a>
    """,
    unsafe_allow_html=True)

with col5:
    st.info("SH 전용 기능")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/update_krxdata" target="_self">
        <div style="
            display: block;
            background-color: #FF9F0A;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">데이터 업데이트</div>
    </a>
    """,
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/manage_remarks" target="_self">
        <div style="
            display: block;
            background-color: #FF9F0A;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">종목 정보 관리</div>
    </a>
    """,
    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <a href="/initialize_krxdata" target="_self">
        <div style="
            display: block;
            background-color: #FF9F0A;
            color: white;
            padding: 0.75em 1em;
            text-align: center;
            border-radius: 12px;
            text-decoration: none;
            font-weight: bold;
            margin-bottom: 0.5em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background-color 0.2s ease;
        ">데이터 초기화</div>
    </a>
    """,
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
