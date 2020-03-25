from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from reddit_visualizer import __version__, __authors__
import sys

packages = find_packages()

def readme():
    with open('README.md') as f:
        return f.read()



class PyTest(TestCommand):
    """
    Custom Test Configuration Class
    Read here for details: https://docs.pytest.org/en/latest/goodpractices.html
    """
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

setup(
    name='reddit_visualizer',
    version=__version__,
    license='GNU GENERAL PUBLIC LICENSE',
    description='Reddit thread visualizer',
    long_description=readme(),
    packages=packages,
    url='',
    author=__authors__,
    author_email='contact@andriymulyar.com',
    keywords='',
    classifiers=[
        '( Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3.5',
        'Natural Language :: English',
        'Topic :: Text Processing :: Linguistic',
        'Intended Audience :: Science/Research'
    ],

    install_requires=[
        'spacy',
        'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz',
        'praw',
        'sklearn',
        'mplcursors'
    ],
    tests_require=["pytest"],
    cmdclass={"pytest": PyTest},
    include_package_data=True,
    zip_safe=False

)