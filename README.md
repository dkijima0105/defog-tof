# Time-of-flight imaging in fog using multiple time-gated exposures

This is a Python implementation of 'Time-of-flight imaging in fog using multiple time-gated exposures.'

## Installation

Use the conda evironment.

```bash
conda env create --name defog --file=environments.yml
```

## Usage


1. Capture data by BEC80T
+ Integration count = 60000.
+ Capture two times: Light on / off.
- Light on: ours
- Light off: ours_amb

2. defog according to demo.ipynb

