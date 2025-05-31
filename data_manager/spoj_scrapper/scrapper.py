import json
import os
import time
import math
import re
import shutil
from dataclasses import asdict
import requests
from bs4 import BeautifulSoup

from data_manager.utils import get_dataset_filepath
from data_manager.spoj_scrapper.scrapper_types import ProblemPreview, Problem

# get this number manually from SPOJ website
SPOJ_PAGES_COUNT = 80

BASE_URL = 'https://www.spoj.com'
PAGE_URL = f'{BASE_URL}/problems/classical'
# PROBLEM_BASE_URL = f'{BASE_URL}/problems'

absolute_path = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_PREVIEW_FILE = f"{absolute_path}/problems_preview.json"
PROBLEMS_FILE = f"{absolute_path}/problems.json"
DATASET_FILE = get_dataset_filepath("scrapper/spoj.json")

PROBLEMS_PER_PAGE = 50

DELAY_BETWEEN_REQUESTS = 1 # seconds

headers = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/136.0.0.0 Safari/537.36'
    ),
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
    ),
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
}


class Scrapper:
    def __init__(self):
        self.current_page_html = None

        self.session = requests.Session()
        self.session.headers.update(headers)

        self.problems_preview = []
        self.problems = []

        self._load_saved_data()

        self.current_page_index = math.ceil(len(self.problems_preview) / PROBLEMS_PER_PAGE)
        self.current_problem_index = len(self.problems)

    def start(self):
        self.request_problems_preview()
        self.request_problems()

        os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
        shutil.copy(PROBLEMS_FILE, DATASET_FILE)

    def request_problems_preview(self):
        while self.current_page_index < SPOJ_PAGES_COUNT:
            print(f"Scraping page {self.current_page_index} / {SPOJ_PAGES_COUNT}")

            try:
                page_url = self._get_current_page_url()
                response = self._request(page_url)
                soup = BeautifulSoup(response.text, 'html5lib')

                tbody = soup.select_one('table.problems > tbody')

                next_problems_preview = self._parse_problems_preview_table(tbody)
            except Exception as e:
                print("Error occurred: when loading and parsing problems preview")
                print(e)
                self._save_and_exit()
            else:
                self.problems_preview.extend(next_problems_preview)
                self.current_page_index += 1

        self._save()

    def request_problems(self):
        while self.current_problem_index < len(self.problems_preview):
            print(f"Scraping problem {self.current_problem_index}/{len(self.problems_preview)}")

            try:
                problem_preview = self.problems_preview[self.current_problem_index]
                response = self._request(problem_preview.url)
                soup = BeautifulSoup(response.text, 'html5lib')

                problem = self._parse_problem(soup, problem_preview)
            except Exception as e:
                problem_id = self.problems_preview[self.current_problem_index].id
                print(f"Error occured: when loading and parsing problem (id={problem_id})")
                print(e)
                self._save_and_exit()
            else:
                self.problems.append(problem)
                self.current_problem_index += 1

            if len(self.problems) % 50: # save every 50 problems parsed, just in case
                self._save()

        self._save()

    def _request(self, url):
        response = self.session.get(url)

        time.sleep(DELAY_BETWEEN_REQUESTS)

        if response.status_code != 200:
            print("Error occurred: status code is not 200")
            self._save_and_exit()

        return response

    def _parse_problems_preview_table(self, tbody):
        problems = []

        for tr in tbody.find_all('tr'):
            problem = ProblemPreview()
            cells = tr.find_all('td')

            problem.id = self._get_text(cells[0])
            problem.title = self._get_text(cells[1])
            problem.url = cells[1].find('a')['href']

            quality_span = cells[2].find('span')

            if quality_span:
                quality_text = cells[2].find('span')['title']
                problem.quality = self._get_text(cells[2].find('span'))
                problem.thumbs_up = quality_text.split(' ')[0][1:]
                problem.thumbs_down = quality_text.split(' ')[1][1:]

            problem.user_count = self._get_text(cells[3])
            problem.acceptance_rate = self._get_text(cells[4])

            problems.append(problem)

        return problems

    def _parse_problem(self, soup, problem_preview):
        problem = Problem()

        # copy values from problem_preview
        problem.id = problem_preview.id
        problem.title = problem_preview.title
        problem.url = problem_preview.url
        problem.quality = problem_preview.quality
        problem.thumbs_up = problem_preview.thumbs_up
        problem.thumbs_down = problem_preview.thumbs_down
        problem.user_count = problem_preview.user_count
        problem.acceptance_rate = problem_preview.acceptance_rate

        # parse tags
        tags_text = self._get_text(soup.select_one('#problem-tags'))
        problem.tags = [] if tags_text == "no tags" else tags_text.split('#')[1:]
        problem.tags = [s.strip() for s in problem.tags]

        problem_body = soup.select_one('#problem-body')
        last_heading = None

        # parse description
        for child in problem_body.children:
            text = self._get_text(child)

            if len(text) == 0:
                continue

            if child.name == 'h3':
                last_heading = self._get_text(child)
                continue

            if last_heading is None:
                problem.description += text + "\n"
            elif last_heading == "Task":
                problem.task_description += text + "\n"
            elif last_heading == "Input":
                problem.input_format += text + "\n"
            elif last_heading == "Output":
                problem.output_format += text + "\n"
            elif last_heading == "Example":
                problem.example = child.text
                break

        # parse meta data
        meta = soup.select_one('#problem-meta')

        for tr in meta.find_all('tr'):
            cells = tr.find_all('td')
            field = self._get_text(cells[0])
            value = self._get_text(cells[1])

            if field == 'Added by:':
                problem.author = value
                problem.author_url = BASE_URL + cells[1].find('a')['href']
            elif field == 'Date:':
                problem.date = value
            elif field == 'Time limit:':
                problem.time_limit = value
            elif field == 'Source limit:':
                problem.source_limit = value
            elif field == 'Memory limit:':
                problem.memory_limit = value
            elif field == 'Cluster:':
                problem.cluster = value
            elif field == 'Languages:':
                problem.languages = value
            elif field == 'Resource:':
                problem.resource = value

        return problem

    def _get_current_page_url(self):
        return f"{PAGE_URL}/sort=0,start={self.current_page_index * 50}"

    def _get_text(self, soup_element):
        text = (soup_element.text
                .replace('\n', ' ')
                .replace('\t', '')
                .replace('\r', '')
                .replace('\u00c2\u00a0', ' ')
                .strip())

        text = re.sub(r'\s+', ' ', text)

        return text

    def _save_and_exit(self):
        self._save()
        exit(1)

    def _save(self):
        with (open(PROBLEMS_PREVIEW_FILE, "w") as f):
            serializable_problems = [asdict(p) for p in self.problems_preview]

            json.dump(serializable_problems, f, indent=4)

        with (open(PROBLEMS_FILE, "w") as f):
            serializable_problems = [asdict(p) for p in self.problems]

            json.dump(serializable_problems, f, indent=4)

    def _load_saved_data(self):
        if os.path.exists(PROBLEMS_PREVIEW_FILE):
            with open(PROBLEMS_PREVIEW_FILE, "r") as f:
                serializable_problems = json.load(f)
                self.problems_preview = [ProblemPreview(**data) for data in serializable_problems]

        if os.path.exists(PROBLEMS_FILE):
            with open(PROBLEMS_FILE, "r") as f:
                serializable_problems = json.load(f)
                self.problems = [Problem(**data) for data in serializable_problems]
