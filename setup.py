from setuptools import setup

setup(
    name="patch_based_inpainting",
    version="0.0.0",
    author="Benjamin Gallois",
    author_email="benjamin@gallois.cc",
    description="",
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
    python_requires='>=3.6',
    zip_safe=False
)
