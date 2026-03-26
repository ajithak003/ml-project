Task 1 — Classify and Handle PII Fields
The dataset contains the following fields:

full_name, email, date_of_birth, zip_code, job_title, diagnosis_notes

Classify each field as either Direct PII or Indirect PII.
For each field, state whether you would drop it, mask it, or pseudonymize it before sharing, and briefly justify your choice.

full_name, email: Direct PII - Drop these columns by default. Mask them only if absolutely necessary. Because these columns directly identify individuals. 

Date_of_birth, zip_code, job_title, diagnosis_notes: Indirect PII - Mask and pseudonymize these columns because they can indirectly identify individuals based on group characteristics, location, or behavioral patterns.


Task 2 — Audit the API Script for Ethical Compliance

import requests

API_URL = "https://healthstats-api.example.com/records"
API_KEY = "free_tier_key_abc123"

records = []
for page in range(1, 101):
    response = requests.get(API_URL, params={"page": page, "key": API_KEY})
    data = response.json()
    records.extend(data["results"])

# Store all records permanently in company database
save_to_database(records)

Identify two ethical or TOS violations present in this script. For each violation, explain what the problem is and suggest a corrected version of the relevant code.

1. **Missing rate limiting**: The script ignores API rate limits, which is an ethical violation. Most APIs (especially free tiers) restrict the number of calls per unit time. Sending 100 consecutive requests can overload their servers and violate their terms of service.

2. **Unauthorized data persistence**: Storing API responses permanently in the company database violates ethical terms. The API provider granted access to view data for a specific purpose, not to copy and permanently store it. This behavior is essentially scraping data without permission or attribution.