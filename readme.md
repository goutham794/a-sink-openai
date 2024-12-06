Credit: [Blogpost by Villoro](https://villoro.com/blog/async-openai-calls-rate-limiter/## Quick Start

throttle-openai allows you to make concurrent OpenAI API calls with automatic rate limiting and structured outputs using Pydantic models.

### Basic Usage

1. First, define your output structure using Pydantic:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Sentiment_Prediction_Output(BaseModel):
    reasoning: str = Field(description="Reasoning for the sentiment prediction in one sentence.")
    sentiment: Literal['Positive', 'Negative', 'Neutral']
```

2. Create a list of inputs with optional IDs for tracking:

```python
input_data = [
    {
        "user_message": "I love this product!",
        "id": "1"  # ID is optional but useful for tracking
    },
    {
        "user_message": "Not satisfied with the quality.",
        "id": "2"
    }
]
```

3. Make concurrent API calls:

```python
from throttle_openai import async_call_open_ai_chat

output, errors = await async_call_open_ai_chat(
    system_prompt="You are a sentiment analyzer. Analyze the sentiment of the given text.",
    gpt_model='gpt-4',
    pydantic_model=Sentiment_Prediction_Output,
    input_data=input_data,
    api_key=your_api_key
)

# Output will be a list of Sentiment_Prediction_Output objects
print(output)

# Check for any errors
if errors:
    print(errors)
```

## Features
- Concurrent Processing: Automatically handles multiple API calls concurrently
- Rate Limiting: Built-in rate limiting to prevent hitting OpenAI's API limits
- Structured Output: Uses Pydantic models for type-safe and validated outputs
- Error Handling: Separate error collection for failed requests
- ID Tracking: Optional ID field to track individual requests and responses