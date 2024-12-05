import asyncio

from loguru import logger

HEADERS = None
RATE_LIMITER_SEMAPHORE = asyncio.Semaphore(25)


def init_openai(secret, n_jobs=None, json=True):
    global HEADERS
    HEADERS = {"Authorization": f"Bearer {secret['api_key']}"}
    if json:
        HEADERS["Content-Type"] = "application/json"

    if n_jobs:
        global RATE_LIMITER_SEMAPHORE
        RATE_LIMITER_SEMAPHORE = asyncio.Semaphore(n_jobs)


def split_valid_and_invalid_records(records, pydantic_model):
    if invalid_results := [x for x in records if not isinstance(x, pydantic_model)]:
        logger.error(f"There are {len(invalid_results)} failed OpenAI calls")
    else:
        logger.info("All calls were successful")

    valid_results = [x for x in records if isinstance(x, pydantic_model)]
    return valid_results, invalid_results