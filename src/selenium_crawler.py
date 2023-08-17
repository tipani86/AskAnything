# Crawler script to take an input website and crawl it for relevant links

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get("http://www.chinatax.gov.cn/chinatax/n810341/n810825/index.html?type=%E8%A7%84%E8%8C%83%E6%80%A7%E6%96%87%E4%BB%B6")
time.sleep(5)
links = []
if os.path.isfile("links.txt"):
    with open("links.txt", "r") as f:
        links = [link.strip() for link in f.readlines()]
i = 0
try:
    while True:
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "tr"))
        )
        for row in rows:
            link = row.find_element(By.TAG_NAME, "a")
            url = link.get_attribute("href")
            if url not in links:
                links.append(url)
        # Check if a link with class "nextbtn" exists
        buttons = driver.find_elements(By.CLASS_NAME, "nextbtn")
        if not buttons:
            break
        time.sleep(1)
        i += 1
        print(f"{i} pages crawled. Number of links found: {len(links)}")
        buttons[0].click()
        # Pause for 2 seconds
        time.sleep(4)
finally:
    # Save the links file
    with open("links.txt", "w") as f:
        for link in links:
            f.write(link + "\n")
    driver.quit()