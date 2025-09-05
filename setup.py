from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    Read the requirements from requirements.txt and return them as a list.
    """
    requirements = []
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.strip() 
                        for req in requirements 
                        if req.strip() 
                        and not req.startswith("#") 
                        and req.strip() != "-e ."
                        ]
    return requirements


setup(
    name="my_package",
    version="0.1.0",
    author="Manish",
    author_email="aryam7842@gmail.com",
    description="A sample Python package",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)