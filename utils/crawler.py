import trafilatura
from trafilatura.settings import DEFAULT_CONFIG
import sys
from time import sleep
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def initialize_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode.
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        # Prevent detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/85.0.4183.102 Safari/537.36'
        })
    except WebDriverException as e:
        print(f"Error initializing WebDriver: {e}", file=sys.stderr)
        driver = None
    return driver



def selenium_crawler(url, timeout = 10):
    driver = initialize_webdriver()
    if not driver:
        return None
    
    driver.implicitly_wait(5)
    try:
        driver.get(url)
        # Wait for the page to load
        WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "body"))
        )
        handle_cookie_popup(driver, timeout=5)
        sleep(1)
        page_source = driver.page_source
        
    except Exception as e:
        print(f"Selenium failed for {url}: {e}")
        page_source = None
        
    finally:
        driver.quit()
        
    return page_source
        


def handle_cookie_popup(driver, timeout=5):
    try:
        # Common button texts; extend this list as needed
        button_texts = ['Accept', 'I Agree', 'Agree', 'Consent', 'Allow All']
        for text in button_texts:
            try:
                cookie_button = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.XPATH, f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"))
                )
                cookie_button.click()
                print(f"Clicked '{text}' button for cookie consent.", file=sys.stderr)
                return True
            except TimeoutException:
                continue
        print("No cookie popup detected or unable to handle.", file=sys.stderr)
    except Exception as e:
        print(f"Error handling cookie popup: {e}", file=sys.stderr)
    return False


def get_page(url):
    page = None
    for i in range(2):
        try: 
            page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
            assert page is not None
        except Exception as e: 
            print(f"Attempt {i+1} with trafilatura failed for {url}: {str(e)}", file=sys.stderr)
   
    if page is None:
        print(f"Using Selenium for {url}", file=sys.stderr)
        page = selenium_crawler(url)
   
    return page


def html2json(page):
    text = trafilatura.extract(page,favor_recall=True, no_fallback=False, output_format="json", with_metadata=True)
    try:
        return json.loads(text)
    except Exception as e:
        print(e)
        return {}


def url2lines(url):
    page = get_page(url)
    if page is None:
        return []
    
    return html2json(page)




