from setuptools import setup

setup(
    name="patch_based_inpainting",
    version="0.0.1",
    author="Analyzable",
    author_email="business@gallois.cc",
    description="Patch-based inpainting.",
    url="https://github.com/Analyzable-FR/patch-based-texture-inpainting",
    packages=['patch_based_inpainting'],
    install_requires=[
        'imageio',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scikit-image',
        'opencv-python'],
    license='MIT',
    python_requires='>=3.7',
    zip_safe=False
)
