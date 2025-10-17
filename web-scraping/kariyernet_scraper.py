import os
import time
import random
import pandas as pd

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from rapidfuzz import fuzz


CSV_PATH = r"/web-scraping/kariyernet_is_ilanlari_URL_listesi.csv"
OUTPUT_EXCEL_PATH = r"/web-scraping/kariyernet_is_ilanlari_detayli.xlsx"
ERROR_LOG_PATH = r"/web-scraping/hatali_kariyernet_linkler.csv"

TEKNOLOJI_KEYWORDS = [
    "teknoloji", "bilgi teknolojileri", "bt", "it", "ict",
    "yazılım", "programcı", "kodlama", "uygulama", "app",

    "developer", "engineer", "mühendis", "architect", "consultant",
    "backend", "frontend", "fullstack", "devops", "qa", "tester",
    "automation", "scrum", "agile", "ci/cd",

    "mobil", "mobile", "android", "ios", "swift", "kotlin", "flutter", "react native",

    "data", "big data", "data scientist", "data analyst", "data engineer",
    "ml", "machine learning", "deep learning", "ai", "artificial intelligence",
    "nlp", "computer vision", "business intelligence", "bi", "etl",

    "cloud", "aws", "azure", "gcp", "kubernetes", "docker",
    "linux", "windows server", "system", "sysadmin", "infrastructure",
    "network", "network engineer", "voip", "telekom", "telecommunication",
    "database", "sql", "nosql", "postgresql", "mysql", "oracle",

    "siber", "cybersecurity", "security", "pentest", "penetration test",
    "ethical hacker", "forensic", "soc analyst", "information security",
    "zero trust", "firewall", "ids", "ips", "encryption",

    "javascript", "typescript", "nodejs", "react", "vue", "angular",
    "php", "laravel", "django", "flask", "spring boot", "dotnet", "c#",
    "golang", "python", "java", "c++", "rust",

    "sap", "erp", "crm", "rpa", "blockchain", "web3", "metaverse",
    "game developer", "unity", "unreal engine", "vr", "ar"
]

def make_driver():
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=tr-TR")
    options.add_argument("--window-size=1280,1000")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver

def click_cookie_if_exists(driver):
    candidates = [
        (By.ID, "onetrust-accept-btn-handler"),
        (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
        (By.XPATH, "//button[contains(., 'Kabul et')]"),
        (By.XPATH, "//button[contains(., 'Accept All')]"),
    ]
    for by, sel in candidates:
        try:
            elem = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((by, sel)))
            elem.click()
            break
        except Exception:
            pass

def get_text_or_none(driver, data_test=None, fallback_selector=None):
    if data_test:
        try:
            el = driver.find_element(By.CSS_SELECTOR, f'[data-test="{data_test}"]')
            txt = el.text.strip()
            if txt:
                return " ".join(txt.split())
        except Exception:
            pass
    if fallback_selector:
        try:
            el = driver.find_element(By.CSS_SELECTOR, fallback_selector)
            txt = el.text.strip()
            if txt:
                return " ".join(txt.split())
        except Exception:
            pass
    return None

def get_text_or_none_description(driver, data_test=None, fallback_selector=None):
    if data_test:
        try:
            el = driver.find_element(By.CSS_SELECTOR, f'[data-test="{data_test}"]')
            txt = el.get_attribute("innerText").strip()
            if txt:
                return " ".join(txt.split())
        except Exception:
            pass
    if fallback_selector:
        try:
            el = driver.find_element(By.CSS_SELECTOR, fallback_selector)
            txt = el.get_attribute("innerText").strip()
            if txt:
                return " ".join(txt.split())
        except Exception:
            pass
    return None

def fetch_job_details(driver, url, first_run=False):
    try:
        driver.get(url)

        if first_run:
            click_cookie_if_exists(driver)

        WebDriverWait(driver, 7).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-test="job-title"]'))
        )

        data = {
            "İlan Linki": url,
            "Pozisyon Başlığı": get_text_or_none(driver, "job-title", "h1"),
            "Şirket Adı": get_text_or_none(driver, "company-name", ".company-name"),
            "Lokasyon": get_text_or_none(driver, "company-location", ".location"),
            "Çalışma Şekli": get_text_or_none(driver, "detail-work-type"),
            "Pozisyon Seviyesi": get_text_or_none(driver, "detail-position-level"),
            "Departman": get_text_or_none(driver, "detail-department-info"),
            "Personel Sayısı": get_text_or_none(driver, "detail-total-count"),
            "Alt Detay": get_text_or_none(driver, "job-sub-detail"),
            "İlan Açıklaması": get_text_or_none_description(driver, "qualifications-and-job-description", ".qualifications-and-job-description"),
        }
        return data, 200
    except Exception as e:
        return None, f"Hata/Selenium: {e}"

def is_technology_job(job_data, threshold=70):
    """Pozisyon Başlığı içinde teknoloji anahtar kelimeleri ile yakın eşleşme kontrolü"""
    title = job_data.get("Pozisyon Başlığı", "").lower()
    if not title:
        return False

    # Anahtar kelimelerle yakınlık skorlarını hesapla
    for keyword in TEKNOLOJI_KEYWORDS:
        score = fuzz.partial_ratio(keyword.lower(), title)
        if score >= threshold:  # eşik değerini %70 seçtik
            return True
    return False

def append_job_to_excel(row_dict, excel_path):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(excel_path):
        df.to_excel(excel_path, index=False)
    else:
        with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            sheet_name = writer.book.active.title
            startrow = writer.book[sheet_name].max_row
            df.to_excel(writer, index=False, header=False, startrow=startrow, sheet_name=sheet_name)

def append_error_to_csv(error_row, csv_path):
    exists = os.path.exists(csv_path)
    pd.DataFrame([error_row]).to_csv(csv_path, mode="a", index=False, header=not exists, encoding="utf-8-sig")

def process_job_urls():
    df_urls = pd.read_csv(CSV_PATH)
    driver = make_driver()

    try:
        total = len(df_urls)
        for idx, row in df_urls.iterrows():
            url = row.get("İlan Linki")
            print(f"{idx + 1}/{total} işleniyor: {url}")

            data, status = fetch_job_details(driver, url, first_run=(idx == 0))
            if status == 200 and data:
                if is_technology_job(data):
                    append_job_to_excel(data, OUTPUT_EXCEL_PATH)
                    print(f"✅ Teknoloji ilanı eklendi: {url}")
                else:
                    print(f"⏭️ Teknoloji ile ilgili değil, atlandı: {url}")
            else:
                append_error_to_csv({"Durum Kodu": status, "URL": url}, ERROR_LOG_PATH)
                print(f"⚠️ Hata: {url} ({status})")

            time.sleep(random.uniform(0.5, 1.2))
    finally:
        driver.quit()
        print(f"🎉 Bitti: {OUTPUT_EXCEL_PATH} | Hata Logu: {ERROR_LOG_PATH}")

if __name__ == "__main__":
    process_job_urls()
