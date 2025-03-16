import time
import logging
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configure logging to display debug messages with timestamps
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# -- Global Setup --
logging.debug("Connecting to MongoDB...")
client = MongoClient("mongodb+srv://test1:test@cluster0.ppljs.mongodb.net/")
db = client["housing_programs"]
profiles_collection = db["profiles"]

def run_selenium_automation(user_full_name: str = "Jay Sid") -> dict:
    """
    Runs the Selenium automation steps for the given user name.
    Returns a dict with 'success' = True/False and optional 'error' message.
    """

    logging.debug(f"Retrieving profile for {user_full_name}...")
    profile = profiles_collection.find_one({"full_name": user_full_name})
    if not profile:
        logging.error(f"User profile for {user_full_name} not found.")
        return {"success": False, "error": f"User '{user_full_name}' not found in DB"}

    logging.debug(f"Profile retrieved: {profile}")

    # Determine the appropriate value for 'applyingNewUnit'
    if profile.get("has_past_due_rent"):
        applying_unit_value = "Assistance with past-due rent"
    elif profile.get("cannot_afford_move_in_costs"):
        applying_unit_value = "Move-in assistance for a unit I have already identified"
    else:
        applying_unit_value = "Both of the above"
    logging.debug(f"Applying unit value determined: {applying_unit_value}")

    # Initialize Selenium WebDriver
    logging.debug("Initializing Selenium WebDriver...")
    driver = webdriver.Chrome()  # Ensure chromedriver is in your PATH

    try:
        # Step 1: Open sferap.com
        logging.debug("Step 1: Opening sferap.com...")
        driver.get("https://sferap.com/")
        time.sleep(5)  # Allow page to load
        logging.debug("Page loaded successfully.")

        # Step 2: Click the "Next" button
        logging.debug("Step 2: Waiting for and clicking the 'Next' button...")
        next_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Next']]"))
        )
        next_button.click()
        time.sleep(5)
        logging.debug("Clicked 'Next' button.")

        # Step 3: Select liveInSF value
        logging.debug("Step 3: Selecting liveInSF value...")
        live_in_sf_elem = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "liveInSF"))
        )
        live_in_sf_select = Select(live_in_sf_elem)
        live_in_sf_value = "yes" if profile.get("current_sf_resident") else "no"
        live_in_sf_select.select_by_value(live_in_sf_value)
        logging.debug(f"Selected liveInSF: {live_in_sf_value}")
        time.sleep(5)

        # Step 4: Select household size
        logging.debug("Step 4: Selecting household size...")
        household_size_elem = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "how-many-in-household"))
        )
        household_size_select = Select(household_size_elem)
        household_size = str(profile.get("household_size", ""))
        household_size_select.select_by_value(household_size)
        logging.debug(f"Selected household size: {household_size}")
        time.sleep(5)

        # Step 5: Enter monthly household income
        logging.debug("Step 5: Entering monthly household income...")
        monthly_income_elem = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "householdMonthlyGross"))
        )
        # Remove disabled attribute if present
        driver.execute_script("arguments[0].removeAttribute('disabled')", monthly_income_elem)
        monthly_income_elem.clear()
        monthly_income = str(profile.get("monthly_household_income", ""))
        monthly_income_elem.send_keys(monthly_income)
        logging.debug(f"Entered monthly income: {monthly_income}")
        time.sleep(5)

        # Step 6: Select applyingNewUnit based on profile data
        logging.debug("Step 6: Selecting applyingNewUnit value...")
        applying_unit_elem = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "applyingNewUnit"))
        )
        applying_unit_select = Select(applying_unit_elem)
        applying_unit_select.select_by_value(applying_unit_value)
        logging.debug(f"Selected applyingNewUnit: {applying_unit_value}")
        time.sleep(10)  # Pause to review the filled form

        return {"success": True}

    except Exception as e:
        logging.exception("An error occurred during the Selenium automation process.")
        return {"success": False, "error": str(e)}
    finally:
        logging.debug("Quitting the Selenium WebDriver.")
        driver.quit()


@app.route('/run-automation', methods=['POST'])
def run_automation():
    """
    API Endpoint to run the Selenium automation for a specific user name.
    Expects JSON with {"full_name": "Jay Sid"} or defaults to "Jay Sid".
    """
    data = request.get_json() or {}
    user_full_name = data.get("full_name", "Jay Sid")

    result = run_selenium_automation(user_full_name)
    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code


if __name__ == '__main__':
    app.run(debug=True, port=5000)
