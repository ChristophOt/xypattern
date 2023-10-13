![codecov](https://codecov.io/gh/CPrescher/xypattern/graph/badge.svg?token=05FUJFOV3R)
![CI](https://github.com/CPrescher/xypattern/actions/workflows/CI.yml/badge.svg)

# xypattern

## Description

A simple small library to handle x-y patterns, such as are collected with x-ray diffraction or Raman spectroscopy. 

## Installation

```bash
pip install xypattern
```

## Usage Examples

### Reading a file
```python
from xypattern import Pattern
import matplotlib.pyplot as plt

p1 = Pattern.from_file('path/to/file')
p1.scaling = 0.5
p1.offset = 0.1

plt.plot(p1.x, p1.y)
plt.show()
```

### Use a background pattern

```python

p2 = Pattern.from_file('path/to/file')
p2.scaling = 0.9
p1.background = p2

```

### Scale and stitch multiple patterns

```python
p1 = Pattern.from_file('path/to/file1')
p2 = Pattern.from_file('path/to/file2')
p3 = Pattern.from_file('path/to/file3')

from xypattern.combine import scale_patterns, stitch_patterns

patterns = [p1, p2, p3]
scale_patterns(patterns)
stitched_pattern = stitch_patterns(patterns)
``` 
