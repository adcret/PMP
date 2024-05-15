# DFXM cell refinement python analysis tool

This repository contains a set of Python scripts and Jupyter notebooks for image processing and cell refinement analysis. The tools provided are designed to facilitate the analysis and visualization of image data.

## Contents

- `image_processor.py`: Contains functions for processing images.
- `scan_functions.py`: Includes various scanning and image analysis functions.
- `cell_refinement_analysis.ipynb`: A Jupyter notebook for performing cell refinement analysis using the provided functions.

## Requirements

To run the scripts and notebooks, the following Python packages are required:

- numpy
- scipy
- matplotlib
- pandas
- scikit-image
- jupyter

You can install the necessary packages using the following command:

```bash
pip install numpy scipy matplotlib pandas scikit-image jupyter
```

## Usage

### Image Processing

The `image_processor.py` script provides functions for processing images. Here’s an example of how to use it:

```python
from image_processor import process_image

# Load and process an image
image_path = 'path/to/your/image.jpg'
processed_image = process_image(image_path)
```

### Scanning Functions

The `scan_functions.py` script includes various functions for scanning and analyzing images. Here’s an example of how to use it:

```python
from scan_functions import scan_image

# Scan and analyze an image
image_path = 'path/to/your/image.jpg'
scan_results = scan_image(image_path)
```

### Cell Refinement Analysis

The `cell_refinement_analysis.ipynb` notebook provides a detailed workflow for cell refinement analysis. To run the notebook:

1. Open the notebook using Jupyter:

```bash
jupyter notebook cell_refinement_analysis.ipynb
```

2. Follow the steps outlined in the notebook to perform the analysis.

   The data can be founf in the data folder.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.
