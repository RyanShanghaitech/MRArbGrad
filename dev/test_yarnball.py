from numpy import *
import mrarbgrad as mag

nPix = 256

if 0:
    # py trajectory
    for tht0 in linspace(0, 2*pi, 100):
        for phi0 in linspace(0, 2*pi, 100):
            print(f"tht0:{tht0}; phi0:{phi0}")
            traj = mag.Yarnball(2*pi/0.5, tht0, phi0)
            grad = traj.gradient()
else:
    # cpp trajectory
    for i in range(1000):
        print(f"test num. {i}")
        mag.scan("Yarnball", nPix=nPix)
        # mag.scan("Spiral", nPix=nPix)
