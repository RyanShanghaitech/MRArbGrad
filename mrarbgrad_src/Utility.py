from numpy import *
from numpy.typing import *
from matplotlib.pyplot import *

def cvtGrad2Traj(arrG:NDArray, dtGrad:int|float, dtADC:int|float, nShift:int|float=1.0) -> tuple[NDArray, NDArray]:
    """
    # description:
    interpolate gradient waveform and calculate trajectory

    # parameter
    `arrG`: array of gradient waveform
    `dtGrad`, `dtADC`: temporal resolution of gradient system and ADC

    # return:
    interpolated trajectory and gradient
    """
    dtShift = nShift*dtADC
    nGrad, nDim = arrG.shape
    nADC = int(dtGrad/dtADC)*(nGrad-1)
    arrG_Resamp = zeros([nADC,nDim], dtype=float64)
    for iDim in range(nDim):
        arrG_Resamp[:,iDim] = interp(dtADC*arange(nADC)+dtShift, dtGrad*arange(nGrad), arrG[:,iDim])
    arrDk = zeros_like(arrG_Resamp)
    arrDk[0,:] = (arrG[0,:] + arrG_Resamp[0,:])*dtShift/2
    arrDk[1:,:] = (arrG_Resamp[:-1] + arrG_Resamp[1:])*dtADC/2
    arrK = cumsum(arrDk,axis=0)
    return arrK, arrG_Resamp

def delGrad(arrG:NDArray, tau:int|float) -> NDArray:
    """
    # description:
    delay the input gradient waveform by time constant tau

    # parameter
    `arrG`: array of single gradient waveform
    `tau`: time constant in RL circuit transfer function

    # return:
    delayed gradient waveform
    """
    assert arrG.ndim == 2, "only single gradient waveform is supported."
    if tau == 0: return arrG.copy() # avoid divided-by-0 later
    nPt, nAx = arrG.shape

    # perform oversample to get better impluse response profile
    ov = clip(10/tau, 1, 1e3).astype(int64) # the smaller the ov, the bigger oversampling is needed
    arrG_ov = zeros([nPt*ov,nAx], dtype=arrG.dtype)
    for iAx in range(nAx):
        arrG_ov[:,iAx] = interp(linspace(0,nPt,nPt*ov,0), linspace(0,nPt,nPt,0), arrG[:,iAx]) # oversample
    nPt *= ov
    tau *= ov

    # derive impluse response of RL circuit
    arrG_Pad = zeros_like(arrG_ov)
    arrG_Pad[:nPt//2,:] = arrG_ov[-1:,:]
    arrG_Pad[nPt//2:,:] = arrG_ov[:1,:]
    arrG_ov = concatenate([arrG_ov, arrG_Pad], axis=0)
    arrT = linspace(0,2*nPt,2*nPt,0) + 0.5
    arrImpResRL = (1/tau)*exp(-arrT/tau)
    if abs(arrImpResRL.sum() - 1) > 1e-2: raise ValueError(f"arrImpResRL.sum() = {arrImpResRL.sum():.2f} (supposed to be 1) (tau too small or too large)")
    
    # perform convolution between input waveform and impulse response
    arrG_ov = fft.ifft(fft.fft(arrG_ov,axis=0)*fft.fft(arrImpResRL)[:,newaxis], axis=0).real
    
    # de-oversample
    arrG = arrG_ov[:nPt:ov,:]

    return arrG

def _walsh(b:float64, k:float64, x:float64) -> float64:
    assert x>=0 and x<1
    
    # Convert k to its base-b representation
    lstKai = []
    while k > 0:
        lstKai.append(k % b)
        k //= b
    lstKai = lstKai[::-1]  # Reverse to get the correct order
    nDig = len(lstKai)
    
    # Convert x to its base-b fractional representation
    lstX = []
    for iDig in range(nDig):
        x *= b
        lstX.append(int(x))
        x -= int(x)
        
    return exp(2*pi*1j*inner(lstKai,lstX)/b)

def calDiaphony(arrX:NDArray, b:float64=2) -> float64: # b-adic diaphony
    assert any(arrX>=0) and any(arrX<1)
    N, s = arrX.shape
    
    nume = 0
    deno = 0
    for vecK in ndindex(*([2] * s)):  # Iterate over all k in [0, 10)^s
        if all(vecK == zeros_like(vecK)): continue  # Skip k = 0

        # Compute weight r_b(k)
        r = prod([b**-floor(log(k+1)/log(b)) if k > 0 else 1 for k in vecK])

        # Compute Walsh coefficient
        meaWalsh = 0
        for vecX in arrX:
            meaWalsh += prod([_walsh(b, k, x) for k, x in zip(vecK, vecX)])
        meaWalsh /= N

        nume += r**2 * abs(meaWalsh)**2
        deno += r**2
    # deno = (1+b)**s - 1 # original implement, abandoned because it doesn't satisfy F_1 = 1

    diaphony = sqrt(nume/deno)
    return diaphony

def rotate(arr:NDArray, ang:float64, axis:int64) -> NDArray:
    if axis==0: # x
        matRot = array([
            [1, 0, 0],
            [0, cos(ang), -sin(ang)],
            [0, sin(ang), cos(ang)],
        ], dtype=float64)
    elif axis==1: # y
        matRot = array([
            [cos(ang), 0, sin(ang)],
            [0, 1, 0],
            [-sin(ang), 0, cos(ang)]
        ], dtype=float64)
    elif axis==2: # z
        matRot = array([
            [cos(ang), -sin(ang), 0],
            [sin(ang), cos(ang), 0],
            [0, 0, 1]
        ], dtype=float64)
    else:
        raise ValueError("axis should be 0, 1, 2 (denotes for x, y, z)")
    
    return arr@matRot.T

def calSphFibPt(nF:int64=250) -> NDArray: # get spherical Fibonacci points
    lstPtFb = []
    for iIntlea in range(nF):
        k = iIntlea - nF/2
        sf = k/(nF//2)
        cf = sqrt(((nF//2)+k)*((nF//2)-k))/(nF//2)
        phi = (1+sqrt(5))/2
        tht= 2*pi*k/phi
        
        xf = cf*sin(tht)
        yf = cf*cos(tht)
        zf = sf
        
        lstPtFb.append(array([xf,yf,zf]))
        
    return array(lstPtFb)

def calJacElip(arrU:NDArray, m:float64) -> tuple[NDArray, NDArray]: # calculate Jacobi elliptic functions sn(u,m) and cn(u,m) numerically
    lstA = [1]
    lstB = [sqrt(1-m)]
    lstC = [0]
    while abs(lstB[-1]-lstA[-1]) > 1e-8:
        aNew = (lstA[-1]+lstB[-1])/2
        bNew = sqrt(lstA[-1]*lstB[-1])
        cNew = (lstA[-1]-lstB[-1])/2
        lstA.append(aNew)
        lstB.append(bNew)
        lstC.append(cNew)
    N = len(lstA) - 1
    lstPhi = [2**N*lstA[N]*arrU]*(N+1)
    for n in range(N,0,-1):
        lstPhi[n-1] = (1/2)*(lstPhi[n] + arcsin(lstC[n]/lstA[n]*sin(lstPhi[n])))
    arrAm = lstPhi[0]
    arrSn = sin(arrAm)
    arrCn = cos(arrAm)
    
    return arrSn, arrCn

def calCompElipInt(m:float64) -> float64: # calculate complete Elliptical integral of the first kind
    lstA = [1]
    lstB = [sqrt(1-m)]
    while abs(lstB[-1]-lstA[-1]) > 1e-8:
        aNew = (lstA[-1]+lstB[-1])/2
        bNew = sqrt(lstA[-1]*lstB[-1])
        lstA.append(aNew)
        lstB.append(bNew)
    return pi/2/lstA[-1]

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
