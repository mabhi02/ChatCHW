from setuptools import setup, find_packages

setup(
    name="matrix-medical",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "pinecone-client>=3.0.0",
        "python-dotenv>=0.19.0",
        "groq>=0.3.0",
        "typing-extensions>=4.0.0"
    ],
    python_requires=">=3.8",
)