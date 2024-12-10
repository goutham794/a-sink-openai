import json
from datetime import datetime
from typing import Optional, List, Any, Dict

import aiohttp
from pydantic import BaseModel
from pydantic import Field
from loguru import logger
import time
import asyncio
import os

import throttle_openai.rate_limiter as rt
import throttle_openai.utils as u
from throttle_openai.tokens import count_tokens, GPTTokens

DEFAULT_MODEL = "gpt-4o-mini"
ENDPOINT_CHAT = "https://api.openai.com/v1/chat/completions"

MAX_REQUESTS_PER_MIN = 10_000
MAX_TOKENS_PER_MIN = 10_000_000


class BaseChatResponse(BaseModel):
    gpt_tokens_used: GPTTokens = Field(
        default_factory=GPTTokens,
        description="Information about the tokens used by the ChatGPT API, "
        "including prompt, completion, and total tokens.",
    )


class BadResponseException(Exception):
    def __init__(self, message="OpenAI API response is malformed or incomplete"):
        super().__init__(message)


def get_json_schema_from_pydantic(pydantic_model):
    """
    To force ChatGPT to output json data.
    """
    schema = pydantic_model.model_json_schema()

    # Manually add "additionalProperties": false to the schema
    schema["additionalProperties"] = False

    json_schema = {
        "name": pydantic_model.__name__,
        "description": pydantic_model.Config.description,
        "schema": schema,
        "strict": True,
    }

    return {"type": "json_schema", "json_schema": json_schema}


async def _call_openai_chat(data, required_tokens):
    await rt.RATE_LIMITER.wait_for_availability(required_tokens)
    async with u.RATE_LIMITER_SEMAPHORE:  # Ensure no more than N tasks run concurrently
        async with aiohttp.ClientSession() as session:
            async with session.post(ENDPOINT_CHAT, headers=u.HEADERS, json=data) as res:
                try:
                    res.raise_for_status()

                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        logger.warning("Rate limit exceeded, retrying")
                        await rt.RATE_LIMITER.wait_for_availability()
                        return await _call_openai_chat(data, required_tokens)
                    else:
                        raise e

                # Update the rate limiter with the response headers
                rt.RATE_LIMITER.update_from_headers(res.headers)

                # Parse the response
                return await res.json()


async def call_openai_chat(
    messages, pydantic_model=None, gpt_model=DEFAULT_MODEL, id=None
):
    required_tokens = count_tokens(messages, model=gpt_model)

    data = {"model": gpt_model, "messages": messages}

    if pydantic_model is not None:
        data["response_format"] = get_json_schema_from_pydantic(pydantic_model)

    response = await _call_openai_chat(data, required_tokens)

    if "usage" not in response or "choices" not in response or not response["choices"]:
        raise BadResponseException(f"Missing expected fields in {response=}")

    usage = response["usage"]
    result = response["choices"][0]["message"]["content"]

    if pydantic_model is None:
        return result, usage

    result_json = json.loads(result)

    class ChatOutput(pydantic_model):
        """
        This class is an extension of the input `pydantic_model`
        It adds multiple metadata meant for debugging and issues resolution
        """

        gpt_called_at: datetime = datetime.now()
        gpt_tokens_used: GPTTokens
        gpt_raw_input: str = json.dumps(messages)
        gpt_model: str = data["model"]
        id: Optional[str]

    return ChatOutput(id=id, gpt_tokens_used=usage, **result_json)

async def async_batch_chat_completion(
    batch_messages: List[Dict[str, Any]], api_key: str = None, pydantic_model=None, gpt_model=DEFAULT_MODEL
):

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "No API key provided and OPENAI_API_KEY environment variable is not set. "
                "Please provide an API key or set the OPENAI_API_KEY environment variable."
            )
        else:
            logger.info(
                "No API key provided - using OPENAI_API_KEY environment variable. "
            )


    t0 = time.monotonic()
    rt.set_rate_limiter(MAX_REQUESTS_PER_MIN, MAX_TOKENS_PER_MIN)

    u.init_openai({'api_key': api_key})

    msg_jobs = f"(n_jobs={u.RATE_LIMITER_SEMAPHORE._value})"
    logger.info(f"Processing {len(batch_messages)} calls to OpenAI asyncronously {msg_jobs}")
    tasks = [
        call_openai_chat(
            messages = item['messages'],
            pydantic_model=pydantic_model,
            gpt_model=gpt_model, 
            id = item.get('id')
        )
        for item in batch_messages
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(
        f"All calls done in {(time.monotonic() - t0)/ 60:.2f} mins {msg_jobs}"
    )
    if pydantic_model is None:
        return results

    output, errors = u.split_valid_and_invalid_records(results, pydantic_model)

    return output, errors