import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/Chemotherapy_regimen"

# Fetch the page content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the first table on the page
table = soup.find('table', {'class': 'wikitable'})

# Extract headers and rows from the table
headers = [header.text.strip() for header in table.find_all('th')]
rows = []
for row in table.find_all('tr')[1:]:
    cells = row.find_all(['td', 'th'])
    rows.append([cell.text.strip() for cell in cells])

# Create a DataFrame and save as CSV
df = pd.DataFrame(rows, columns=headers)
df.to_csv("chemotherapy_regimens.csv", index=False)

print("Table saved as 'chemotherapy_regimens.csv'")

