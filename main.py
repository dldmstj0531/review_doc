from src.report_generator import generate_reports

if __name__ == "__main__":
    marketing, service = generate_reports("data/adjectives_with_service_ratings.csv")

    with open("marketing_report.txt", "w", encoding="utf-8") as f:
        f.write(marketing)

    with open("service_report.txt", "w", encoding="utf-8") as f:
        f.write(service)

    print("리포트 생성 완료! marketing_report.txt / service_report.txt 확인하세요.")
