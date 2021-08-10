import setuptools


# DO NOT CHANGE: required by the s2agemaker template,
s2agemaker_requirements = [
    "gunicorn",
    "uvicorn[standard]",
    "pydantic",
    "fastapi",
    "click",
    "python-json-logger",
]

# Add your python dependencies
model_requirements = [
    "numpy",
    "sklearn"
]

dev_requirements = ["pytest", "mypy", "black", "requests", "types-requests"]

setuptools.setup(
    name="s2-fos",
    version="0.0.1",
    description="S2's paper Field of Study classifier",
    url="https://github.com/allenai/s2-fos/",
    packages=setuptools.find_packages(),
    install_requires=s2agemaker_requirements + model_requirements,
    extras_require={"dev": dev_requirements},
    python_requires=">3.7,<3.9",
)
