# src/utils/api_utils.py

import requests
from requests.exceptions import RequestException

from src.utils.rate_limiter import rate_limited_api_call, handle_rate_limit

@rate_limited_api_call
def call_api_with_retry(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={'accept': 'application/json'}, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if handle_rate_limit(response, attempt):
                continue
            elif response.status_code == 400:
                print(f"Bad request for URL: {url}")
                return None
            else:
                raise
    raise Exception("Max retries reached")