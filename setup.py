# this is responsible for using our model as a package

from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    ''' returns list of requirements '''
    
    requirements=list()
    with open(file_path) as file_object:
        requirements=file_object.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
      name='Image captioning',
      version='0.0.1',
      author='Deepraj Mazumder',
      author_email='deeprajmazumder11@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements('requirement.txt')
      )