import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page to scrape
url = "https://www.cancer.gov/about-cancer/treatment/drugs"

# Fetch the page content
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful
soup = BeautifulSoup(response.text, 'html.parser')

# Initialize lists to store generic and brand names
generic_names = []
brand_names = []

# Find all drug entries on the page
for drug in soup.find_all('li'):
    text = drug.get_text(strip=True)
    if '(' in text and ')' in text:
        # Extract brand and generic names
        brand, generic = text.split('(', 1)
        generic = generic.rstrip(')')
        brand_names.append(brand.strip())
        generic_names.append(generic.strip())
    else:
        # No brand name, only generic name
        brand_names.append("")
        generic_names.append(text.strip())

# Create a DataFrame
data = {
    'Generic Name': generic_names,
    'Brand Name': brand_names
}
df = pd.DataFrame(data)

# Save the data to a CSV file
df[df['Brand Name'].fillna('').str.strip().apply(len)>0].to_csv('cancer_drug_mapping.csv', index=False)

print("CSV file 'cancer_drug_mapping.csv' has been created successfully.")

