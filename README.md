# xinterp

This package enables exact round to even datetime64 interpolation.

## Installation

Clone the repository, enter the folder and:

```
pip install -e .
```

## Usage

```python
import numpy as np
from xinterp import interp_intlike

xp = np.array([0, 10, 20])
fp = np.array([0, 1000, 2000], dtype="datetime64[s]")
x = np.array([-5, 0, 5, 10, 15, 20, 25])

nat = np.datetime64("NaT")
f = interp_intlike(x, xp, fp, left=nat, right=nat)
f_expected = np.array([nat, 0, 500, 1000, 1500, 2000, nat], dtype="datetime64[s]")
assert np.array_equal(f, f_expected, equal_nan=True)
assert f.dtype == f_expected.dtype
```