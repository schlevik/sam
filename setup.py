from setuptools import setup, find_packages

setup(
    name='mrc-stresstest',
    version='1.0.0',
    description='Workbench to generate mrc stress tests',
    url='http://github.com/schlevik/mrc-stresstest',
    author='Viktor Schlegel',
    author_email='viktor.schlegel@manchester.ac.uk',
    license='GPLv3',
    packages=find_packages(),
    zip_safe=False,
    setup_requires=["nose"],
    tests_require=["nose", "coverage"]
)
