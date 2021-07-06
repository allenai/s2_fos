import setuptools


runtime_requirements = [
    "uvicorn[standard]",
    "pydantic",
    "fastapi"
]

dev_requirements = [
    "pytest",
    "mypy",
    "black"
]

setuptools.setup(
    name="your-model-name-here",
    version="0.0.1",
    description="Describe your model here",
    url="https://github.com/allenai/s2-model-template/",
    packages=setuptools.find_packages(),
    install_requires=runtime_requirements,
    python_requires=">3.7,<3.9"
)
