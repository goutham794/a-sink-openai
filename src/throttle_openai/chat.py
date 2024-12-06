import json
from datetime import datetime
from typing import Optional

import aiohttp
from pydantic import BaseModel
from pydantic import Field
from loguru import logger
import time
import asyncio

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


def _construct_messages(prompt, text=None):
    messages = []
    if text is not None:
        messages += [{"role": "system", "content": prompt}]

    messages += [{"role": "user", "content": text or prompt}]
    return messages


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
    prompt, text=None, pydantic_model=None, gpt_model=DEFAULT_MODEL, id=None
):
    messages = _construct_messages(prompt, text)
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
        gpt_raw_input: str = text or prompt
        gpt_model: str = data["model"]
        id: Optional[str]

    return ChatOutput(id=id, gpt_tokens_used=usage, **result_json)

async def async_call_open_ai_chat(
    prompt, input_data, api_key, pydantic_model=None, gpt_model=DEFAULT_MODEL
):
    t0 = time.monotonic()
    rt.set_rate_limiter(MAX_REQUESTS_PER_MIN, MAX_TOKENS_PER_MIN)

    u.init_openai({'api_key': api_key})

    msg_jobs = f"(n_jobs={u.RATE_LIMITER_SEMAPHORE._value})"
    logger.info(f"Processing {len(input_data)} calls to OpenAI asyncronously {msg_jobs}")
    tasks = [
        call_openai_chat(
            prompt, pydantic_model=pydantic_model, gpt_model=gpt_model, **row
        )
        for row in input_data
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(
        f"All calls done in {(time.monotonic() - t0)/ 60:.2f} mins {msg_jobs}"
    )
    if pydantic_model is None:
        return results

    output, errors = u.split_valid_and_invalid_records(results, pydantic_model)

    return output, errors