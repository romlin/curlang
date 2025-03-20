from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="curlang",
    version="0.0.12",
    license="Apache-2.0",
    packages=find_packages(),
    package_data={
        'curlang': ['resources/python_executor.py'],
    },
    include_package_data=True,
    python_requires=">=3.10.0, <3.13.0",
    install_requires=[
        "aiofiles==24.1.0",
        "beautifulsoup4==4.13.3",
        "croniter==6.0.0",
        "cryptography==44.0.2",
        "fastapi==0.115.11",
        "hnswlib==0.8.0",
        "httpx==0.28.1",
        "itsdangerous==2.2.0",
        "lark==1.2.2",
        "ngrok==1.4.0",
        "psutil==7.0.0",
        "pydantic==2.10.6",
        "pypdf==5.4.0",
        "python-multipart==0.0.20",
        "requests==2.32.3",
        "rich==13.9.4",
        "sentence-transformers==3.4.1",
        "toml==0.10.2",
        "uvicorn[standard]==0.34.0",
        "zstandard==0.23.0"
    ],
    author="Henric Romlin",
    author_email="henric@romlin.com",
    description="The Robot Language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/romlin/curlang",
    entry_points={
        "console_scripts": [
            "curlang=curlang.main:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only"
    ]
)
