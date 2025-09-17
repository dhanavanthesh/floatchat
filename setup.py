"""
FloatChat Setup Script
Production-ready ARGO oceanographic data analysis system
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="floatchat",
    version="1.0.0",
    author="FloatChat Team",
    description="AI-powered conversational interface for ARGO ocean data discovery and visualization",
    long_description="""
    FloatChat is a production-ready system for analyzing oceanographic data from ARGO floats.
    It provides natural language querying capabilities powered by AI, real-time data processing,
    and interactive visualizations for CTD and BGC parameters.

    Features:
    - Natural language queries for oceanographic data
    - Real-time ARGO NetCDF data processing
    - MongoDB storage with geospatial indexing
    - ChromaDB vector search for enhanced AI context
    - Interactive visualizations (Plotly, Folium)
    - Groq AI integration with RAG pipeline
    - Support for BGC parameters (oxygen, chlorophyll, nitrate)
    """,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "floatchat=app:main",
        ],
    },
    package_data={
        "": ["*.txt", "*.md", "*.yaml", "*.yml"],
    },
)