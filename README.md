# Non-Cartesian MRI Gradient Waveform Toolbox (MRArbGrad)
Python library for designing the gradient waveforms for arbitrary MRI trajectories.

The trajectories can be specified as a Python function, or as a set of k-space points. A trajectory library is also provided. The underlying C/C++ code can be ported to compatible pulse sequence projects for fast gradient waveform design.

## How to use
### Install
```bash
$ pip install mrarbgrad
```

### Import Libraries
```python
from numpy import *
import mrarbgrad as mag
```

### Define Your Trajectory Function
```python
def Rosette(t):
    rho = 0.5 * sin(5*pi*t)
    phi = 3*pi*t
    return rho*array([cos(phi), sin(phi)])
```

### Solve for the Gradient Waveform
```python
grad = mag.solve(Rosette, pmin=0, pmax=1)
```
`pmin` and `pmax` define the bounds of the function parameter `t`. `grad[:,0]`, `grad[:,1]`, and `grad[:,2]` are the x-, y-, and z-axis gradient waveforms, respectively.

*For more usages such as specifying the hardware constraints, use of the trajectory library and other utilities, please refer to the [Examples](https://github.com/rui-luo1002/MRArbGrad/tree/main/example).*

## Acknowledgements
The algorithm in this library is proposed in:

> [1] Luo R, Huang H, Miao Q, Xu J, Hu P, Qi H. Real-Time Gradient Waveform Design for Arbitrary k-Space Trajectories. IEEE Transactions on Biomedical Engineering. 2026 Oct;73(10):3491-502. doi:10.1109/TBME.2026.3654117

