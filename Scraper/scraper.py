from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import cv2

def init_driver():
    chrome_options = Options()
    driver = webdriver.Chrome(chrome_options)
    return driver

def scrape(driver, site: str):
    driver.get(site)
    driver.save_screenshot('screenshot.png')
    print(driver.get_window_size())
    image = cv2.imread('screenshot.png')
    with open("Scraper/scrape.js") as f:
        elements = driver.execute_script(f.read())
    for element in elements['buttons']:
        cv2.rectangle(image, (int(element['x']), int(element['y'])), (int(element['x']+element['width']), int(element['y']+element['height'])), (0, 0, 255), 3)

    cv2.imshow("lol", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    driver.close()

if __name__ == '__main__':
    driver = init_driver()
    with open("Scraper/websites.txt") as f:
        sites = f.readlines()
    for site in sites:
        scrape(driver, site)
