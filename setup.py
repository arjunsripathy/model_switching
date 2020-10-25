from setuptools import setup

setup(
    name="model_switching",
    install_requires=[
        "gym",
        "flake8",  # linter
        "numpy",
        "matplotlib",
        "moviepy",
        "pyglet>=1.4",  # visualization
        "tensorflow==2.1.0",
    ],
    version='1.0',
    author='Arjun Sripathy',
    author_email='arjunsripathy@berkeley.edu',
    description='Model Switching Experiments with InterACT lab self-driving simulator',
    license='MIT'
)
