from setuptools import setup, find_packages

setup(
    name="my_llm_tool",
    version="0.1",
    packages=find_packages(),
    install_requires=["torch", "transformers", "python-dotenv", "bitsandbytes"],
)
