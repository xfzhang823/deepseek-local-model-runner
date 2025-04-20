from setuptools import setup, find_packages

setup(
    name="DeepSeek Local Model Runner",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["torch", "transformers", "python-dotenv", "bitsandbytes"],
)
