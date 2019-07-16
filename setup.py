import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    requirements = req.readlines()
    print(requirements)

setuptools.setup(
    name='pymlalgo',
    version='0.0.1',
    author='Vivek Kumar',
    author_email='vivekuma@uw.edu',
    description='Implementation of standard machine learning algorithms from scratch in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vivekkr12/machine-learning-algorithms',
    packages=setuptools.find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ]
)
