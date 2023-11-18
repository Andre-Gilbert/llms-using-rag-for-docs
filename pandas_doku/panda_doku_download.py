import requests
import os
from bs4 import BeautifulSoup

base_url = "https://pandas.pydata.org/docs/reference/api/"
endings_file_path = os.path.join(os.path.dirname(__file__), 'endings.txt')

with open(endings_file_path, 'r') as endings_file:
    endings = [line.strip() + '.html' if not line.strip().endswith('.html') else line.strip() for line in endings_file]


output_folder = os.path.join(os.path.dirname(__file__), 'curled')

for ending in endings:
    url = f"{base_url}{ending}"
    output_file = f"{output_folder}{ending.replace('.', '_')}.html"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Successful: {url}")
        
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

output_folder = os.path.join(os.path.dirname(__file__), 'textfiles')

input_folder = os.path.join(os.path.dirname(__file__), 'curled')

for filename in os.listdir(input_folder):
    if filename.endswith('.html') or filename.endswith('.htm'):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # <article class="bd-article" role="main">
        main_content = soup.find('article', {'class': 'bd-article'})

        # Extract
        if main_content:
            text_content = main_content.get_text()

            output_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_filename, 'w', encoding='utf-8') as output_file:
                output_file.write(text_content)

            print(f"Processed: {filename}")
        else:
            print(f"Error {filename}")

print("done")
