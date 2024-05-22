import os
import requests
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Create a folder named 'CSE_3C' if it doesn't exist
output_folder = 'CSE_3C'
os.makedirs(output_folder, exist_ok=True)

# List of usernames and passwords
usernames_passwords = [
    'JIS/2021/0396',
'JIS/2021/0218',
'JIS/2021/1299',
'JIS/2021/0524',
'JIS/2021/0905',
'JIS/2021/0449',
'JIS/2021/0358',
'JIS/2021/0348',
'JIS/2021/0349'
# 'JIS/2021/0294',
# 'JIS/2021/0552',
# 'JIS/2021/0496',
# 'JIS/2021/0443',
# 'JIS/2021/0549',
# 'JIS/2021/0882',
# 'JIS/2021/0572',
# 'JIS/2021/0362',
# 'JIS/2021/0491',
# 'JIS/2021/0657',
# 'JIS/2021/0895',
# 'JIS/2021/0908',
# 'JIS/2021/0519',
# 'JIS/2021/0724',
# 'JIS/2021/0194',
# 'JIS/2021/0218',
# 'JIS/2021/0623',
# 'JIS/2021/0727',
# 'JIS/2021/0887',
# 'JIS/2021/0275',
# 'JIS/2021/0879',
# 'JIS/2021/0356',
# 'JIS/2021/0441',
# 'JIS/2021/0146',
# 'JIS/2021/0256',
# 'JIS/2021/0677',
# 'JIS/2021/0873',
# 'JIS/2021/0313',
# 'JIS/2021/0619',
# 'JIS/2021/0278',
# 'JIS/2021/0927',
# 'JIS/2021/0423',
# 'JIS/2021/0681',
# 'JIS/2021/0297',
# 'JIS/2021/0933',
# 'JIS/2021/1299',
# 'JIS/2021/0389',
# 'JIS/2021/0407',
# 'JIS/2021/0536',
# 'JIS/2021/0499',
# 'JIS/2021/0725',
# 'JIS/2021/0412',
# 'JIS/2021/1224',
# 'JIS/2022/0044',
# 'JIS/2022/0584',
# 'JIS/2022/0518',
# 'JIS/2022/0009',
# 'JIS/2022/0633',
# 'JIS/2022/0590',
# 'JIS/2022/0649',
# 'JIS/2022/0578',
# 'JIS/2022/0541',
# 'JIS/2022/0669',
# 'JIS/2022/0576',
# 'JIS/2022/0647',
# 'JIS/2022/0581',
# 'JIS/2022/0572',
# 'JIS/2022/0525',
# 'JIS/2022/0526',
# 'JIS/2022/0512',
# 'JIS/2022/0024',
# 'JIS/2020/0620'
]

for username_password in usernames_passwords:
    username = username_password
    password = username_password

    print(f'Logging in as user {username}...')

    driver = webdriver.Edge()
    driver.get('https://jisgroup.net/erp/forms/frmcommonlogin.aspx')

    try:
        # Find the username and password input fields and the login button by their HTML attributes
        username_field = driver.find_element(By.ID, 'ctxt_user_name')
        password_field = driver.find_element(By.ID, 'ctxt_pass_word')

        # Find the dropdown element by its HTML attribute
        dropdown = Select(driver.find_element(By.ID, 'cmd_ligin_type'))
        dropdown.select_by_visible_text('Student')

        # Enter the login credentials
        username_field.send_keys(username)
        password_field.send_keys(password)

        # Click the login button
        script = "document.querySelector('.btn.btn-primary.btn-block.waves-effect.waves-light.text-white').click();"
        driver.execute_script(script)

        # Use an explicit wait to wait for some expected condition (e.g., page navigation)
        wait = WebDriverWait(driver, 100)
        wait.until(EC.url_to_be('https://jisgroup.net/erp/forms/frmCommonDashBoard.aspx'))

        # Find the <img> element by its ID
        img_element = driver.find_element(By.ID, 'img_profile_pict')
        img_src = img_element.get_attribute('src')

        # Download the image using Python requests and save it in the 'secB' folder with the username as the filename
        image_filename = os.path.join(output_folder, f"{username.replace('/', '_')}.jpeg")
        response = requests.get(img_src)

        if response.status_code == 200:
            with open(image_filename, 'wb') as f:
                f.write(response.content)
            print(f'Image downloaded successfully for user {username}.')
        else:
            print(f'Failed to download image for user {username}.')

    except Exception as e:
        print(f'Error for user {username}: {str(e)}')

    finally:
        # Close the WebDriver
        driver.quit()
