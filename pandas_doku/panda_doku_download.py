import requests
import os
from bs4 import BeautifulSoup

base_url = "https://pandas.pydata.org/docs/reference/api/"
endings_file_path = os.path.join(os.path.dirname(__file__), 'endings.txt')

with open(endings_file_path, 'r') as endings_file:
    endings = [line.strip() + '.html' if not line.strip().endswith('.html') else line.strip() for line in endings_file]

output_parent_folder = os.path.join(os.path.dirname(__file__), 'curled')
textfiles_parent_folder = os.path.join(os.path.dirname(__file__), 'textfiles')

# Counter for folder naming
folder_counter = 1

for ending in endings:
    url = f"{base_url}{ending}"

    # Determine output folder based on the counter
    output_folder = os.path.join(output_parent_folder, f'curled{folder_counter}')
    textfiles_folder = os.path.join(textfiles_parent_folder, f'textfiles{folder_counter}')

    # Create folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(textfiles_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f"{ending.replace('.', '_')}.html")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Successful: {url}")

            # Check if we need to increment the folder counter
            if len(os.listdir(output_folder)) >= 999:
                folder_counter += 1

    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Reset the folder counter for textfiles
folder_counter = 1

for filename in os.listdir(output_parent_folder):
    if filename.startswith('curled'):
        input_folder = os.path.join(output_parent_folder, filename)

        for filename in os.listdir(input_folder):
            if filename.endswith('.html') or filename.endswith('.htm'):
                file_path = os.path.join(input_folder, filename)
                print(f"Processing: {file_path}")

                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()

                soup = BeautifulSoup(html_content, 'html.parser')

                main_content = soup.find('article', {'class': 'bd-article'})

                if main_content:
                    text_content = main_content.get_text()

                    textfiles_folder = os.path.join(textfiles_parent_folder, f'textfiles{folder_counter}')
                    os.makedirs(textfiles_folder, exist_ok=True)

                    output_filename = os.path.join(textfiles_folder, f"{os.path.splitext(filename)[0]}.txt")
                    with open(output_filename, 'w', encoding='utf-8') as output_file:
                        output_file.write(text_content)

                    print(f"Processed: {filename}")

                    # Check if we need to increment the folder counter
                    if len(os.listdir(textfiles_folder)) >= 999:
                        folder_counter += 1
                else:
                    print(f"Error {filename}")

print("done")
