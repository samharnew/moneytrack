from setuptools import setup, find_packages

setup(
    name="moneytrack",
    packages=find_packages(),
    scripts=["scripts/build.sh"],
    package_data={
        'sample_data': [
            'balance_updates.csv',
            'accounts.csv',
            'transfers.csv',
            'data.xlsx',
        ],
    },
    # get the version from the python tag
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        "scipy>=1.4.1",
        "numpy>=1.18.1",
        "pandas>=0.25.3",
        "matplotlib>=3.0.0",
        "xlrd>=1.0.0",
    ],
    # metadata to display on PyPI
    author="Sam Harnew",
    author_email="sam.harnew@gmail.com",
    description="This package helps to track personal finances spread over many accounts",
    keywords="personal finance tracking money",
    url="https://github.com/samharnew/moneytrack",
    project_urls={
        "Source Code": "https://github.com/samharnew/moneytrack",
    },
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ]
)
