from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (NoSuchElementException)
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import pytz
import time

def get_stock_twits_twits(symbol):
    base_url = f"https://stocktwits.com/symbol/{symbol}"
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)

   
    driver.get(base_url)
    
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "RichTextMessage_body__Fa2W1")))
    except NoSuchElementException:
        print("No Element found as in the input")
        driver.quit()
    twit_list = []
    bigTwitList = []
    twit_times = []
    twit_datetime =[]
    bigTwitDatetime = []
    runner = True
    count = 0
    utc = pytz.UTC
    f = open("text.txt", "w")
    f.close()

    now = datetime.now(timezone.utc)

    end_date = now.strftime("%Y-%m-%d %H:%M:%S")
    

    
    start_date = (now - timedelta(days=365))
    
    while runner:
        f= open("text.txt","a")
        f.write("\nhi")
        f.close()
        time.sleep(5)
        twit_list.clear()
        twit_datetime.clear()
        twit_times.clear()
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "RichTextMessage_body__Fa2W1")))
        html = driver.page_source

        soup = BeautifulSoup(html, "html.parser")
        
        twit_list = soup.find_all('div', class_='RichTextMessage_body__Fa2W1')
        twit_times = soup.find_all('time')
        if(utc.localize(datetime.strptime(twit_times[-1]['datetime'], "%Y-%m-%dT%H:%M:%SZ"))>=start_date):
            f = open("text.txt","a")
            f.write("\n"+utc.localize(datetime.strptime(twit_times[-1]['datetime'], "%Y-%m-%dT%H:%M:%SZ")).strftime('%Y-%m-%d %H:%M:%S'))
            f.close()
            count=0
        else:
            f = open("text.txt","a")
            f.write("\nDONE\n"+utc.localize(datetime.strptime(twit_times[-1]['datetime'], "%Y-%m-%dT%H:%M:%SZ")).strftime('%Y-%m-%d %H:%M:%S')+ "\nDONE\n")
            f.close()
            count+=1
            if(count == 3):
                twit_list = twit_list[:len(twit_times)]
                break
 

    for (twit,twitTime) in zip(twit_list, twit_times):
        f = open("text.txt","a")
        f.write("\n"+twit.text)
        x=utc.localize(datetime.strptime(twitTime['datetime'], "%Y-%m-%dT%H:%M:%SZ"))
        f.write("\n"+x.strftime('%Y-%m-%d %H:%M:%S'))
        f.write("\n******************************")
        f.close()
    driver.quit()

symbol = 'AAPL'
get_stock_twits_twits(symbol)


