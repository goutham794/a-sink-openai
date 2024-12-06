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
            "user_message": review[1],
            "id": str(review[0]) # ID is not mandatory, but it's useful for tracking the output
            })

    output, errors = await async_call_open_ai_chat(system_prompt=system_prompt,
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
        (1, "I like this product. It's good."),
        (2, "I don't like this product. It's bad."),
        (3, "This product is okay."),
        (4, "I've been using this product for a while and I'm satisfied with it."),
        (5, "I regret buying this product. It's a complete disaster."),
        (6, "The product is good, but it could be better."),
        (7, "I've had a great experience with this product. Highly recommended!"),
        (8, "I'm not sure how I feel about this product. It has some good points, but also some bad points."),
        (9, "I've used this product for a month and I'm still not satisfied. I'm returning it."),
    ]

    asyncio.run(analyze_sentiment(sentences, system_prompt))