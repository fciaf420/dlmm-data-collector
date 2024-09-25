# src/utils/rate_limiter.py

import time
import random
from functools import wraps
from ratelimit import limits, sleep_and_retry

from config.settings import API_CALLS_LIMIT, API_CALLS_PERIOD, MAX_BACKOFF_TIME, INITIAL_BACKOFF_TIME

def exponential_backoff(attempt):
    backoff_time = min(MAX_BACKOFF_TIME, INITIAL_BACKOFF_TIME * (2 ** attempt) + random.uniform(0, 1))
    return backoff_time

def rate_limited_api_call(func):
    @wraps(func)
    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_CALLS_PERIOD)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def handle_rate_limit(response, attempt):
    if response.status_code == 429 or (500 <= response.status_code < 600):
        sleep_time = exponential_backoff(attempt)
        print(f"Rate limit hit or server error. Retrying in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
        return True
    return False