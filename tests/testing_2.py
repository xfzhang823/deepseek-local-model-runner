def run_llm_task(task, **kwargs):
    print(f"Task: {task}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")


# Create a dictionary of keyword arguments
params = {"text": "Hello, world!", "target_lang": "French", "max_length": 1000}

# Call run_llm_task, unpacking the dictionary into keyword arguments.
run_llm_task("translation", **params)
