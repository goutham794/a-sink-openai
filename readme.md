# throttle-openai

A Python library for making concurrent OpenAI API calls with automatic rate limiting and structured outputs using Pydantic models.
Credit: [Blogpost by Villoro](https://villoro.com/blog/async-openai-calls-rate-limiter/)

## Installation

```bash
pip install throttle-openai
```

## Quick Start

### 1. Define Your Output Structure

First, create a Pydantic model that defines your expected output structure:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Sentiment_Prediction_Output(BaseModel):
    reasoning: str = Field(description="Reasoning for the sentiment prediction in one sentence.")
    sentiment: Literal['Positive', 'Negative', 'Neutral']

    class Config:
        title = "Sentiment Prediction Output"
        description = "Sentiment prediction with reasoning."
```

### 2. Make API Calls

Here's a complete example showing how to analyze sentiments for multiple product reviews:

```python
from throttle_openai import async_batch_chat_completion
import asyncio
import os

async def analyze_sentiment():
    # Prepare batch messages
    batch_messages = [
        {
            "messages": [
                {"role": "system", "content": "You are a sentiment analyzer. Analyze the sentiment of the given text."},
                {"role": "user", "content": "I like this product. It's good."}
            ],
            "id": "1"  # id is optional
        },
        {
            "messages": [
                {"role": "system", "content": "You are a sentiment analyzer. Analyze the sentiment of the given text."},
                {"role": "user", "content": "I don't like this product. It's bad."}
            ],
            "id": "2"
        }
    ]

    # Make concurrent API calls
    output, errors = await async_batch_chat_completion(
        batch_messages=batch_messages,
        gpt_model='gpt-4o-mini',
        pydantic_model=Sentiment_Prediction_Output,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    return output, errors

# Run the analysis
output, errors = asyncio.run(analyze_sentiment())

# Process results
print("Results:", output)
if errors:
    print("Errors:", errors)
```



## Features
- Concurrent Processing: Automatically handles multiple API calls concurrently
- Rate Limiting: Built-in rate limiting to prevent hitting OpenAI's API limits
- Structured Output: Uses Pydantic models for type-safe and validated outputs
- Error Handling: Separate error collection for failed requests
- ID Tracking: Optional ID field to track individual requests and responses

## Environment Variables
Set your OpenAI API key as an environment variable:
```
export OPENAI_API_KEY=your_api_key
```

## Complete Example
Check out `examples/basic_usage_structured_output.py` for a complete example featuring structured outputs.