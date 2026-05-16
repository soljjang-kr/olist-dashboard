"""
마크다운 리포트를 스타일이 적용된 HTML 리포트로 변환하는 스크립트입니다.

주요 기능:
- `markdown` 패키지를 이용해 마크다운을 HTML로 변환합니다.
- 전문적인 CSS 스타일을 적용하여 웹 브라우저에서 보기 좋게 렌더링합니다.
- `Project1/report/eda_report_professional.html`로 저장합니다.
"""

import os
import markdown

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(BASE_DIR, 'report', 'eda_report_professional.md')
HTML_PATH = os.path.join(BASE_DIR, 'report', 'eda_report_professional.html')

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report</title>
    <style>
        body {{
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1, h2, h3, h4, h5 {{
            color: #2c3e50;
            margin-top: 1.5em;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f4f6f7;
            color: #333;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        blockquote {{
            margin: 0;
            padding: 10px 20px;
            background-color: #e8f4f8;
            border-left: 5px solid #3498db;
            color: #34495e;
        }}
        p {{
            margin-bottom: 1em;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

def convert_md_to_html():
    if not os.path.exists(MD_PATH):
        print(f"Error: {MD_PATH} 를 찾을 수 없습니다.")
        return

    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 확장 기능 추가: 표(tables), 코드블록 등
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    final_html = HTML_TEMPLATE.format(content=html_content)

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"HTML 리포트가 성공적으로 생성되었습니다: {HTML_PATH}")

if __name__ == "__main__":
    convert_md_to_html()
