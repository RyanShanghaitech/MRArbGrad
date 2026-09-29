import mrarbgrad as mag
from numpy import *
from matplotlib.pyplot import *

pathDs = "/mnt/c/Users/13287/Downloads/260126/Grad/"

lstArrK0 = mag.loadF32(f"{pathDs}M0PE.hdr", f"{pathDs}M0PE.bin")
print(lstArrK0)
lstArrK0 = [k0*0 for k0 in lstArrK0]
mag.saveF32(f"{pathDs}M0PE.hdr", f"{pathDs}M0PE.bin", lstArrK0)
print(lstArrK0)
