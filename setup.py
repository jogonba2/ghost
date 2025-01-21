from typing import Dict
from setuptools import setup, find_packages

VERSION: Dict[str, str] = {}
with open("ghost/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    version=VERSION["VERSION"],
    name="symanto-ghost",
    description="Ghost adaptative thresholding",
    author="Symanto Research GmbH",
    author_email="jose.gonzalez@symanto.com",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    python_requires=">=3.7.0",
    zip_safe=False,
)
