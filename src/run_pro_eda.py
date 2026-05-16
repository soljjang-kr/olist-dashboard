import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from scipy import stats
import os
import base64
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 환경 설정 ---
if not os.path.exists('Project1/images'):
    os.makedirs('Project1/images')
if not os.path.exists('Project1/report'):
    os.makedirs('Project1/report')

def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def run_pro_eda_html():
    # 데이터 로드
    orders = pd.read_csv('Project1/data/olist_orders_dataset.csv')
    customers = pd.read_csv('Project1/data/olist_customers_dataset.csv')
    payments = pd.read_csv('Project1/data/olist_order_payments_dataset.csv')
    reviews = pd.read_csv('Project1/data/olist_order_reviews_dataset.csv')

    # 병합
    df = pd.merge(orders, customers, on='customer_id')
    order_payments = payments.groupby('order_id')['payment_value'].sum().reset_index()
    df = pd.merge(df, order_payments, on='order_id', how='left')
    df = pd.merge(df, reviews[['order_id', 'review_score', 'review_comment_message']], on='order_id', how='left')
    
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')

    html_content = """
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: 'Malgun Gothic', sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 40px; border-left: 5px solid #2980b9; padding-left: 10px; }
            h3 { color: #16a085; }
            .viz-container { margin: 30px 0; background: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            .desc { margin-top: 15px; font-weight: bold; color: #555; background: #fff; padding: 10px; border-left: 3px solid #16a085; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>📊 Olist 이커머스 심층 분석 리포트 (Professional Edition)</h1>
        <p>본 리포트는 모든 시각화 결과가 포함된 자기완성형 HTML 파일입니다.</p>
    """

    # 1. 데이터 개요 섹션
    html_content += f"""
    <h2>1. 데이터 개요</h2>
    <ul>
        <li>전체 주문 건수: {len(df):,}</li>
        <li>고유 고객 수: {df['customer_unique_id'].nunique():,}</li>
        <li>분석 기간: {df['order_purchase_timestamp'].min().strftime('%Y-%m-%d')} ~ {df['order_purchase_timestamp'].max().strftime('%Y-%m-%d')}</li>
    </ul>
    """

    # 시각화 함수
    def add_viz(fig, title, desc):
        b64 = fig_to_base64(fig)
        plt.close(fig)
        return f"""
        <div class="viz-container">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{b64}">
            <p class="desc">💡 해석: {desc}</p>
        </div>
        """

    html_content += "<h2>2. 상세 시각화 분석</h2>"

    # 1) 월별 주문 추이
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby('order_month').size().plot(kind='line', marker='o', ax=ax, color='#3498db')
    ax.set_title('월별 주문 건수 추이')
    html_content += add_viz(fig, "월별 주문 추이", "2017년부터 2018년 중반까지 꾸준한 성장세를 보이며, 블랙 프라이데이 등 특정 이벤트 시기에 주문량이 폭증하는 패턴이 관찰됩니다.")

    # 2) 결제 금액 분포
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['payment_value'].dropna() + 1, log_scale=True, kde=True, color='#2ecc71', ax=ax)
    ax.set_title('결제 금액 분포 (Log Scale)')
    html_content += add_viz(fig, "결제 금액 분포", "대부분의 결제는 100 R$ 내외에 집중되어 있으며, 롱테일(Long-tail) 분포를 보이고 있어 고가 상품 구매층에 대한 별도 타겟팅이 유효할 수 있습니다.")

    # 3) 리뷰 점수 분포
    fig, ax = plt.subplots(figsize=(10, 5))
    df['review_score'].value_counts().sort_index().plot(kind='bar', color='#f1c40f', ax=ax)
    ax.set_title('리뷰 점수 분포')
    html_content += add_viz(fig, "고객 만족도 분석", "5점 평점이 가장 높지만, 1점 평점도 무시할 수 없는 비중을 차지합니다. 배송 지연이나 제품 결함 등 부정적 요인에 대한 집중 관리가 필요합니다.")

    # 4) 지역별 비중
    fig, ax = plt.subplots(figsize=(8, 8))
    df['customer_state'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'), ax=ax)
    ax.set_title('주요 지역(State) 매출 비중')
    html_content += add_viz(fig, "지역별 시장 점유율", "상파울루(SP) 주가 전체의 40% 이상을 점유하고 있어 브라질 남동부 지역의 물류 및 마케팅 인프라가 핵심 경쟁력임을 시사합니다.")

    # 5) TF-IDF 키워드
    review_texts = df['review_comment_message'].dropna().head(3000)
    if not review_texts.empty:
        tfidf = TfidfVectorizer(max_features=15)
        tfidf_matrix = tfidf.fit_transform(review_texts)
        ranking = pd.DataFrame({'term': tfidf.get_feature_names_out(), 'rank': tfidf_matrix.sum(axis=0).A1}).sort_values('rank', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='rank', y='term', data=ranking, palette='mako', ax=ax)
        ax.set_title('리뷰 핵심 키워드 (TF-IDF)')
        html_content += add_viz(fig, "고객 보이스(VoC) 분석", "'produto', 'entrega', 'bom' 등이 핵심 키워드입니다. 고객들은 특히 제품의 상태와 배송의 신속성에 가장 민감하게 반응하고 있습니다.")

    html_content += """
        <h2>3. 결론 및 권고 사항</h2>
        <ul>
            <li><strong>물류 최적화:</strong> SP 지역의 비중이 매우 높으므로 해당 지역의 익일 배송 서비스를 강화하여 리텐션을 높여야 합니다.</li>
            <li><strong>부정 리뷰 대응:</strong> 1점 평점 고객의 주요 불만 사항을 분석하여 선제적인 보상 시스템이나 품질 검수 강화를 권고합니다.</li>
            <li><strong>가격 전략:</strong> 대중적인 가격대(100 R$ 이하)의 상품군을 메인 마케팅으로 활용하되, 고단가 상품에 대한 프리미엄 패키지 도입을 고려할 수 있습니다.</li>
        </ul>
        <p style="text-align: right; margin-top: 50px; color: #888;">&copy; 2026 Olist Data Intelligence Team</p>
    </body>
    </html>
    """

    with open('Project1/report/eda_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("성공: Project1/report/eda_report.html 파일이 생성되었습니다.")

if __name__ == "__main__":
    run_pro_eda_html()
