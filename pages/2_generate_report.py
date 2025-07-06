import streamlit as st
import pandas as pd
from src.report_generator import generate_reports
import tempfile

st.set_page_config(page_title="리포트 생성", page_icon="📝")
st.title("GPT 기반 마케팅 리포트 생성")

# CSV 파일이 세션에 있는지 확인
if "uploaded_file" not in st.session_state:
    st.warning("메인 페이지에서 CSV 파일을 먼저 업로드해주세요.")
    st.stop()

uploaded_file = st.session_state["uploaded_file"]
uploaded_file.seek(0)

# Streamlit에서 바로 읽은 업로드 파일은 generate_reports에 쓸 수 없기 때문에, 임시 저장 필요
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(st.session_state["uploaded_file"].read())
    tmp_path = tmp.name

if st.button("리포트 생성하기"):
    with st.spinner("GPT-4o로 리포트 생성 중..."):
        try:
            marketing_report, service_report = generate_reports(tmp_path)

            st.success("리포트 생성 완료!")
            st.subheader("마케팅 전략 리포트")
            st.text_area("Marketing Report", marketing_report, height=400)
            st.download_button("⬇다운로드", marketing_report, file_name="marketing_report.txt")

            st.subheader("서비스 개선 전략 리포트")
            st.text_area("Service Report", service_report, height=400)
            st.download_button("⬇다운로드", service_report, file_name="service_report.txt")

        except Exception as e:
            st.error(f"오류 발생: {e}")