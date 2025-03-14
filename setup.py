from setuptools import  find_packages , setup
from typing import List

def fun(filename : str) -> List[str]:
    requirements = []
    with open(filename) as obj:
        requirements = obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements


setup(
    name = 'student-performance',
    version = '0.0.1',
    author = 'Uday Khunt',
    author_email = 'udaykhunt02@gmail.com',
    packages = find_packages(),
    install_requires = fun('requirements.txt')
)