import pandas as pd
from src.gpt_client import get_report_from_gpt

def load_reviews(file_path: str):
    df = pd.read_csv(file_path)
    pos_reviews = df[df["Recommended"] == "yes"]["Adjectives/Adverbs"].dropna().tolist()
    neg_reviews = df[df["Recommended"] == "no"]["Adjectives/Adverbs"].dropna().tolist()
    return pos_reviews, neg_reviews

def build_prompt(reviews: list[str], report_type: str):
    sample = "\n".join(f"- {r}" for r in reviews[:20])  # 상위 20개만 사용
    if report_type == "marketing":
        return f"""다음은 고객의 긍정 리뷰입니다. 아래 내용을 기반으로 마케팅 전략 리포트를 작성해주세요:\n\n{sample}"""
    else:
        return f"""다음은 고객의 부정 리뷰입니다. 아래 내용을 기반으로 서비스 개선 전략 리포트를 작성해주세요:\n\n{sample}"""

def generate_reports(file_path: str):
    pos_reviews, neg_reviews = load_reviews(file_path)
    pos_prompt = build_prompt(pos_reviews, "marketing")
    neg_prompt = build_prompt(neg_reviews, "service")

    marketing_report = get_report_from_gpt(pos_prompt)
    service_report = get_report_from_gpt(neg_prompt)

    return marketing_report, service_report