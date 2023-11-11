import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pathlib import Path

class WebScraper():
    def fetch_pdfs(output:bool=True) -> None:
        curr_path = os.path.curdir

        url = 'https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-collision-reports/'
        location = Path(curr_path) / 'reports'

        if not os.path.exists(location):
            os.mkdir(location)

        soup = BeautifulSoup(requests.get(url).text, 'html.parser')

        links = soup.select("div[class$='wp-block-dmv-container shadow-box']")
        soup2 = BeautifulSoup(str(links[0]), 'html.parser')

        count = 0
        for link in soup2.select("a[href*='portal']"):
            slash = os.path.sep

            if link['href'][-1] == slash:
                link['href'] = link['href'][:-1]
            file_end = link['href'].split(slash)[-1] + '.pdf'
            print('Downloading... {}'.format(location / file_end))

            filename = Path(location) / file_end
            if not os.path.exists(filename):
                with open(filename, 'wb') as f:
                    f.write(requests.get(urljoin(url,link['href'])).content)
                count += 1

        print('{} files downloaded!'.format(count))