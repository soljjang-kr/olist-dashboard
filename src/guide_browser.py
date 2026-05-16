from playwright.sync_api import sync_playwright
import os

def capture_streamlit_cloud():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Streamlit Cloud 로그인 페이지 접속
        print("Streamlit Community Cloud 접속 중...")
        page.goto('https://share.streamlit.io/')
        page.wait_for_load_state('networkidle')
        
        # 스크린샷 저장
        screenshot_path = 'streamlit_cloud_start.png'
        page.screenshot(path=screenshot_path)
        print(f"스크린샷 저장 완료: {os.path.abspath(screenshot_path)}")
        
        browser.close()

if __name__ == "__main__":
    capture_streamlit_cloud()
