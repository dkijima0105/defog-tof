# Time-of-flight imaging in fog using multiple time-gated exposures

This is a Python implementation of 'Time-of-flight imaging in fog using multiple time-gated exposures.'
Our algorithm can work for any short pulse time-of-flight camera, but we assume the use of BEC80T by brookman tecknology in the code.

## Installation

Use the conda evironment.

```bash
conda env create --name defog --file=environments.yml
```

## Usage


1. Capture data by BEC80T
+ Integration count = 60000.
+ Capture two times: Light on / off for ambient removal.
+ Data format: .csv file (set average mode when you save the data on a GUI program.)

2. defog them according to demo.ipynb

