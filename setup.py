from setuptools import setup, find_packages

setup(
    name='dsClass',
    version='1.0.27',
    packages=find_packages(),
    description='A useful module',
    author='Guy',
    author_email='example@example.com',
    install_requires=[
          'pydotplus',
      ],
    include_package_data=True,
    package_data={'': ['*.csv','*.pickle','*.ipynb','*.mp4','*.pyc','*.npy','*.data-00000-of-00001','*.index']},
)
