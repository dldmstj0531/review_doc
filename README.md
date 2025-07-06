# ReviewDoctor
리뷰닥터 레포지토리입니다.

## Streamlit 
- stream 설치하기
  `pip install streamlit`

- steramlit이 잘 설치되었는지 확인
  `streamlit hello`

- streamlit로 파이썬 실행하기
  `streamlit run (파일명).py`

---

## 프로젝트트 디렉토리 구조

ReviewDoctor/
├── data/ # 샘플 리뷰 CSV 파일
│ └── adjectives_with_service_ratings.csv
├── pages/ # Streamlit 멀티 페이지 구성
│ ├── 1_review_upload_and_analysis.py # 더미 분석 + 시각화
│ └── 2_generate_report.py # GPT 기반 리포트 생성
├── src/ # GPT 호출 및 리포트 처리 로직
│ ├── gpt_client.py # Azure OpenAI 연결
│ └── report_generator.py # 프롬프트 생성 및 결과 반환
├── streamlit_app.py # 메인 페이지 (CSV 업로드 및 라우팅 안내)
├── main.py # CLI 기반 GPT 리포트 생성 진입점
├── .env # 실제 실행용 환경변수 (로컬)
├── .env.example # 공유용 환경변수 템플릿
├── requirements.txt # 의존성 패키지 목록
└── README.md

> `streamlit_app.py`에서 CSV 파일을 업로드하면 세션을 통해 모든 페이지에서 공유됩니다.

---

## .env 설정

루트 디렉토리에 `.env` 파일을 생성하고 다음 정보를 입력하세요:

```env
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://smu-3team.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=smu-3team-gpt-4o-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview

---

## 실행 방법

1. 의존성 설치
`pip install -r requirements.txt`

2. Streamlit 앱 실행
`python -m streamlit run streamlit_app.py`

3. 웹 브라우저에서 자동 실행되는 페이지에서 사용
- 기본 주소: http://localhost:8501