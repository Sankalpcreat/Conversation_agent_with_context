# utils/memory_handler.py

import tiktoken
from langchain_core.messages import trim_messages as core_trim_messages, HumanMessage, SystemMessage

# Define a token counter using tiktoken
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a given text using the specified model's tokenizer.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def trim_messages(messages, max_tokens=2000, model_name="gpt-3.5-turbo"):
    """Trim messages to keep within the max token limit."""
    total_tokens = 0
    trimmed_messages = []

    for message in reversed(messages):
        message_tokens = count_tokens(message.content, model_name)
        if total_tokens + message_tokens > max_tokens:
            break
        trimmed_messages.insert(0, message)
        total_tokens += message_tokens

    return trimmed_messages

def summarize_history(messages, model):
    """Summarize the chat history into a single message."""
    # Create a prompt to instruct the model to summarize
    summary_prompt = SystemMessage(
        content="Summarize the key points of the previous conversation in a concise manner."
    )

    # Include the messages to be summarized and the summary prompt
    response = model.invoke(messages + [summary_prompt])

    # Return the summary as a system message
    return SystemMessage(content=response.content)
