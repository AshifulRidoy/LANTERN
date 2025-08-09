from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agentic-hate-speech-detection",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="Multi-Task Hate Speech Detection & Explainable Counter-Speech Generation with DeBERTa + LLaMA 3 + Agentic LangChain Orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/agentic-hate-speech-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
        ],
        "serving": [
            "vllm>=0.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-hate-speech=training_script:main",
            "serve-api=deployment_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.yml", "*.yaml", "*.json"],
    },
)