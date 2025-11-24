# Automated Deal Sourcing Pipeline & Database

## Project Overview
This tool automates the lead generation process for private equity and search funds. It replaces the manual workflow of browsing industry directories with an automated pipeline that:
1.  Extracts company data (Revenue, Location, Industry).
2.  Filters targets based on investment criteria (e.g., Revenue > $2M).
3.  Persists qualified leads into a SQL database for CRM integration.

## Impact: Reduces deal sourcing time by ~90% by filtering out unqualified targets before human review.

## Note on "Simulation Mode"
**For public demonstration purposes, the live Selenium web driver has been replaced with a `simulate_scraping_run()` function using mock data.**

This is done to:
1.  Prevent violating the Terms of Service (ToS) of proprietary business directories.
2.  Protect the privacy of real business owners' contact information.

## Production Implementation Logic
In the production environment, the `simulate_scraping_run` function is replaced with the live Selenium logic below:

```python
# Production Logic Snippet (Selenium)
def real_scraping_run(target_url):
    options = Options()
    options.add_argument("--headless") # Runs without opening browser window
    driver = webdriver.Chrome(options=options)
    
    driver.get(target_url)
    time.sleep(2) # Wait for DOM load
    
    # Locate all company cards
    listings = driver.find_elements(By.CLASS_NAME, "company-listing-row")
    
    for listing in listings:
        name = listing.find_element(By.CSS_SELECTOR, ".biz-name").text
        revenue_raw = listing.find_element(By.XPATH, "//span[@id='revenue']").text
        # ... validation and data cleaning logic ...
