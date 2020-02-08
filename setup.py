import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="xai-metrics",
    version="0.0.2",
    author="Grzegorz Huk",
    author_email="mrgreg557@gmail.com",
    description="A package for analysis and evaluating metrics for Explainable AI (XAI)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greghvk/ml_models_explainability",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='xai, explainability, ai, machine learning',
    license="MIT",
    install_requires=["lime>=0.1",
                      "matplotlib>=3.1",
                      "numpy>=1.18",
                      "pandas>=1.0",
                      "seaborn>=0.10",
                      "shap>=0.34",
                      "scikit-learn>=0.22"],
    python_requires='>=3',
)
