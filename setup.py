from setuptools import find_packages, setup

HYPHEN_E_DOT="-e ."
def get_required_packages(file_path):
    required_packages=[]
    with open(file_path) as file:
        required_packages=file.readlines()

        if HYPHEN_E_DOT in required_packages:
            required_packages.remove(HYPHEN_E_DOT)
    return required_packages


setup(
    name="lrproject",
    version="0.0.1",
    author="Awol",
    packages=find_packages(),
    install_requires=get_required_packages("requirements.txt")
)