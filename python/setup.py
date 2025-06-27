# setup.py  (located in ~/Desktop/Biennale_2025/python/)
from setuptools import setup, find_packages

setup(
	name="sztempl",
	version="0.1.0",
	packages=find_packages(),  # find sztempl/ automatically
	install_requires=[
		"picamera2", "numpy", "opencv-python", "pygame", "PyYAML"
	],
	entry_points={
		"console_scripts": [
			"sztempl=sztempl.core:main",  # command → module:function
		],
	},
)
