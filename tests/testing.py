from contextlib import contextmanager
import time


@contextmanager
def llm_stream_context(message):
    print("Starting LLM streaming...")
    yield mock_llm_stream(message)  # Yield the generator function
    print("\n[Stream closed] Cleaning up resources...")


def mock_llm_stream(message):
    words = message.split()
    for word in words:
        yield word
        time.sleep(1.5)


# Using 'with' to stream LLM output
with llm_stream_context("Hello, this is a simulated LLM response!") as stream:
    for word in stream:
        print(f"Received: {word}", end=" ", flush=True)
