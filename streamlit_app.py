import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import openai
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from dotenv import load_dotenv

# -----------------------------------
# 1) .env 환경변수 로드
# -----------------------------------
load_dotenv()

ML_ENDPOINT    = os.getenv("ML_ENDPOINT")
ML_PRIMARY_KEY = os.getenv("ML_PRIMARY_KEY")

AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

openai.api_type    = "azure"
openai.api_base    = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key     = AZURE_OPENAI_API_KEY

# -----------------------------------
# 2) Azure ML 호출 함수
# -----------------------------------
def call_azure_ml(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    1) 원본 DataFrame → JSON 페이로드 ({"data":[…]})
    2) Azure ML 호출 → JSON 응답({ "csv_data": "...CSV 문자열..."})
    3) "csv_data" 필드를 판독하여 DataFrame으로 반환
    """
    # 1) Inf → NaN → None(→JSON으로 보낼 때 null)
    df = df_input.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    payload = {"data": records}

    # 디버그용 로그: 전송 전 payload 확인(레코드 수만 출력)
    print(">>> Sending payload to Azure ML:")
    print(f"  ▶ 총 {len(records)}개 레코드를 전송합니다.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ML_PRIMARY_KEY}"
    }

    # 2) Azure ML 호출
    response = requests.post(ML_ENDPOINT, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    result_json = response.json()

    # 디버그용 로그: 받은 JSON 출력
    print("===== Azure ML returned JSON =====")
    print(result_json)

    # 3) "csv_data" 키가 있으면 DataFrame으로 변환하여 반환
    if "csv_data" in result_json:
        csv_text = result_json["csv_data"]
        # StringIO로 CSV 파싱
        return pd.read_csv(io.StringIO(csv_text))
    else:
        raise RuntimeError(f"Unexpected response format: {result_json}")

# -----------------------------------
# 3) Azure OpenAI 호출 함수 (필요 시 사용)
# -----------------------------------
def call_azure_openai(df_result: pd.DataFrame) -> str:
    """
    Azure ML 결과 DataFrame을 GPT 프롬프트로 보내고, 생성된 리포트 문자열 반환
    """
    csv_buffer = io.StringIO()
    df_result.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()

    prompt = f"""
다음은 Azure ML 분석 결과 CSV입니다.
이 데이터를 바탕으로 두 가지 리포트를 작성해주세요.

1) 📈 마케팅 전략 리포트  
   - 고객 세그먼트별 마케팅 제안  
2) 🛠️ 서비스 개선 전략 리포트  
   - 서비스 항목별 개선 인사이트  

아래는 CSV 전체 내용입니다:
---
{csv_text}
---"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in writing business reports based on customer 리뷰 데이터."},
        {"role": "user", "content": prompt}
    ]

    completion = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        max_tokens=1500,
        temperature=0.7
    )
    return completion.choices[0].message.content

# -----------------------------------
# 4) Streamlit 설정 및 UI
# -----------------------------------
st.set_page_config(page_title="Review Report Generator", page_icon="🛫", layout="wide")
st.title("리뷰 기반 리포트 생성기")

# 4.1) CSV 업로드 위젯 (원본 데이터)
uploaded_file = st.file_uploader("📥 원본 리뷰 CSV 파일 업로드", type=["csv"])
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("✅ 원본 CSV 업로드 완료! 사이드바 메뉴를 선택하세요.")
        st.session_state["df_raw"] = df_raw
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")
        st.stop()
else:
    st.info("먼저 리뷰 원본 CSV 파일을 업로드해주세요.")
    st.stop()

# 4.2) 사이드바 메뉴
menu = st.sidebar.selectbox("🔍 기능 선택", (
    "리뷰 분석",
    "GPT 리포트 생성"
))

# -----------------------------------
# 5) "리뷰 분석" 페이지
# -----------------------------------
if menu == "리뷰 분석":
    st.header("🔍 1. 리뷰 분석 (Azure ML 호출)")

    # df_raw가 세션에 없으면 업로드부터 다시 안내
    if "df_raw" not in st.session_state:
        st.error("원본 CSV를 업로드해야 합니다.")
        st.stop()

    df_raw = st.session_state["df_raw"]

    # ML 호출 버튼
    if st.button("🔄 Azure ML 분석 실행"):
        with st.spinner("Azure ML 앤드포인트 호출 중..."):
            try:
                df_result = call_azure_ml(df_raw)
                st.session_state["df_result"] = df_result
                st.success("✅ Azure ML 분석 완료!")
            except Exception as e:
                st.error(f"Azure ML 호출 오류: {e}")
                st.stop()
    else:
        st.info("“🔄 Azure ML 분석 실행” 버튼을 눌러 분석을 시작하세요.")

    # 시각화: df_result가 있으면 기존 시각화 함수 호출
    if "df_result" in st.session_state:
        st.subheader("📊 분석 결과 시각화")
        df_result = st.session_state["df_result"]

        # 예시 시각화: DataFrame 미리보기
        st.write("#### 결과 데이터 미리보기")
        st.dataframe(df_result.head(5))

        # 예시: OverallRating 분포
        if "OverallRating" in df_result.columns:
            st.write("##### OverallRating 분포")
            counts = df_result["OverallRating"].value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(counts.index.astype(str), counts.values, color="skyblue")
            ax.set_xlabel("OverallRating")
            ax.set_ylabel("건수")
            ax.set_title("OverallRating 분포")
            st.pyplot(fig)

        # 예시: 워드클라우드 (ClusterID별 키워드 시각화 예시)
        if "Nouns" in df_result.columns:
            st.write("##### 명사 워드클라우드 예시")
            # 모든 행의 Nouns 문자열을 합쳐서 워드클라우드 생성
            all_nouns = " ".join(df_result["Nouns"].astype(str).tolist())
            if len(all_nouns.strip()) > 0:
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_nouns)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                ax_wc.imshow(wordcloud, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

# -----------------------------------
# 6) "GPT 리포트 생성" 페이지
# -----------------------------------
elif menu == "GPT 리포트 생성":
    st.header("📝 2. GPT 리포트 생성")

    if "df_result" not in st.session_state:
        st.warning("먼저 ‘리뷰 분석’ 메뉴에서 Azure ML 분석을 완료해주세요.")
        st.stop()

    df_result = st.session_state["df_result"]

    # 리포트 생성 버튼
    if st.button("🖋️ 리포트 생성"):
        with st.spinner("GPT-4o-mini 호출 중..."):
            try:
                report_text = call_azure_openai(df_result)
                st.session_state["report_text"] = report_text
                st.success("✅ GPT 리포트 생성 완료!")
            except Exception as e:
                st.error(f"GPT 호출 오류: {e}")
                st.stop()
    else:
        st.info("“🖋️ 리포트 생성” 버튼을 눌러주세요.")

    # 생성된 리포트가 있으면 미리보기 + 다운로드
    if "report_text" in st.session_state:
        st.subheader("📋 생성된 리포트")
        st.text_area("## Generated Report", st.session_state["report_text"], height=400)
        st.download_button(
            label="⬇️ 리포트 다운로드 (TXT)",
            data=st.session_state["report_text"],
            file_name="generated_report.txt",
            mime="text/plain"
        )
