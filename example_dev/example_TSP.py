from numpy import *
from numpy.typing import *
from matplotlib.pyplot import *
from scipy.stats import qmc
from python_tsp.heuristics import solve_tsp_local_search as solve_tsp

def genTspTraj(nCity:int) -> NDArray:
    # print("# 1. Generate random k-space points (the cities for the TSP)")
    arrCity = empty([nCity,3], dtype=double)
    arrCity[:,:2] = qmc.Halton(d=2).random(n=nCity)-0.5
    arrCity[0,:] = 0
    arrCity[:,-1] = 0

    # print("# 2. Calculate the distance matrix between all points")
    matDist = norm(arrCity[:,newaxis,:] - arrCity[newaxis,:,:], axis=-1)
    matDist[:, 0] = 0

    # print("# 3. Solve the TSP to get the optimal order (permutation)")
    idxSort, _ = solve_tsp(matDist, 0)
    return arrCity[idxSort]

def rmCity(arrCity:NDArray, angMax:double=pi/6, distMin:double=1) -> NDArray:
    print(arrCity.shape)
    while 1:
        nCity = arrCity.shape[0]
        lstIdxRm = []
        for iCity in range(1,nCity-1):
            vec0 = arrCity[iCity-1,:] - arrCity[iCity,:]
            vec0Norm = vec0/norm(vec0)
            vec1 = arrCity[iCity+1,:] - arrCity[iCity,:]
            vec1Norm = vec1/norm(vec1)
            if inner(vec0Norm, vec1Norm)>cos(angMax) or norm(vec0)<distMin:
                lstIdxRm.append(iCity)
        if len(lstIdxRm)>0: arrCity = delete(arrCity, lstIdxRm, axis=0)
        else: break
    
    return arrCity

from numpy.linalg import norm

def intpCity(arrCity:NDArray, nPix:int) -> NDArray:
    arrCity_Intp = []
    for i in range(len(arrCity) - 1):
        k0 = arrCity[i]
        k1 = arrCity[i+1]
        
        # Generate points along the segment from start_k to end_k
        nIntpStep = int(nPix*norm(k1-k0))
        for iStep in range(nIntpStep):
            t = iStep / nIntpStep
            cityIntp = (1 - t) * k0 + t * k1
            arrCity_Intp.append(cityIntp)
    return array(arrCity_Intp)