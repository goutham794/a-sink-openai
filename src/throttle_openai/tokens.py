import tiktoken
from pydantic import BaseModel
from pydantic import Field

MODEL_GPT_4 = "gpt-4"
MODEL_TOKENIZER_DEFAULT = MODEL_GPT_4


class GPTTokens(BaseModel):
    prompt_tokens: int = Field(
        0, description="The number of tokens used in the prompt."
    )
    completion_tokens: int = Field(
        0, description="The number of tokens used in the completion."
    )
    total_tokens: int = Field(
        0, description="The total number of tokens used (prompt + completion)."
    )


def count_tokens(messages, model=MODEL_TOKENIZER_DEFAULT):
    """Estimate the number of tokens used by the messages."""

    # I belive the tokenization does not change between sub models
    if model.startswith(MODEL_GPT_4):
        model = MODEL_GPT_4

    encoding = tiktoken.encoding_for_model(model)

    tokens_per_message = 3  # A rough estimate for the start of each message
    tokens_per_name = 1  # If names are used, they add an extra token

    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
            if key == "name":
                total_tokens += tokens_per_name

    # Adding 3 tokens for the overall start and end of the message
    total_tokens += 3

    return total_tokens