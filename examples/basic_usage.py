from throttle_openai import async_call_open_ai_chat
from pydantic import BaseModel, Field
from typing import List, Literal
from textwrap import dedent
import asyncio
import os


class Sentiment_Prediction_Output(BaseModel):
    reasoning: str = Field(description="Reasoning for the sentiment prediction in one sentence.")
    sentiment: Literal['Positive', 'Negative', 'Neutral']

    class Config:
        title = "Sentiment Prediction Output"
        description = "Sentiment prediction with reasoning."


async def analyze_sentiment(reviews: List[str], system_prompt: str) -> Sentiment_Prediction_Output:

    input_data = []
    for review in reviews:
        input_data.append(
            {
                "text": review
                    })

    output, errors = await async_call_open_ai_chat(prompt=system_prompt,
                                                   gpt_model = 'gpt-4o-mini',
                                                   pydantic_model=Sentiment_Prediction_Output,
                                                   input_data=input_data,
                                                   api_key=os.getenv("OPENAI_API_KEY_WF"))

    print(output)
    # Print errors if any
    if errors:
        print(errors)


if __name__ == "__main__":

    system_prompt = dedent("""\
        You are a helpful and honest assistant for analyzing sentiments.
        You are given a product review and you need to predict the sentiment of the review.
        You should also provide a reasoning for your prediction.
        """)

    sentences = [
        "I love this product! It's amazing.",
        "This is the worst product I've ever used.",
        "The product is okay, nothing special.",
        "I've been using this product for a while and I'm satisfied with it.",
        "I regret buying this product. It's a complete disaster.",
        "The product is good, but it could be better.",
        "I've had a great experience with this product. Highly recommended!",
        "I'm not sure how I feel about this product. It has some good points, but also some bad points.",
        "I've used this product for a month and I'm still not satisfied. I'm returning it.",
    ]

    asyncio.run(analyze_sentiment(sentences, system_prompt))