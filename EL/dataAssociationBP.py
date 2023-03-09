import numpy as np

def dataAssociationBP(legacy, new=None, checkConvergence=10, threshold=1e-6, numIterations=100):
    m, n = legacy.shape[0] - 1, legacy.shape[1]
    assocProbNew = np.ones((m,))
    assocProbExisting = np.ones((m+1,n))
    messagelhfRatios = np.ones((m,n))
    messagelhfRatiosNew = np.ones((m,))
    
    if n == 0 or m == 0:
        return assocProbExisting, assocProbNew, messagelhfRatios, messagelhfRatiosNew
    
    if new is None:
        new = 1
    
    om = np.ones((1,m))
    on = np.ones((1,n))
    muba = np.ones((m,n))
    for iteration in range(numIterations):
        mubaOld = muba

        prodfact = muba * legacy[1:,:]
        sumprod = legacy[0,:] + np.sum(prodfact, axis=0)

        normalization = (sumprod[om,:] - prodfact)
        # hard association if message value is very large
        normalization[normalization == 0] = np.finfo(float).eps
        muab = legacy[1:,:] / normalization
        summuab = new + np.sum(muab, axis=1)
        normalization = summuab[:,on] - muab
        # hard association if message value is very large
        normalization[normalization == 0] = np.finfo(float).eps
        muba = 1 / normalization

        if iteration % checkConvergence == 0:
            distance = np.max(np.abs(np.log(muba/mubaOld)))
            if distance < threshold:
                break
    
    assocProbExisting[0,:] = legacy[0,:]
    assocProbExisting[1:,:] = legacy[1:,:] * muba
    
    for target in range(n):
        assocProbExisting[:,target] /= np.sum(assocProbExisting[:,target])
    
    messagelhfRatios = muba
    assocProbNew = new / summuab
    
    messagelhfRatiosNew = np.concatenate((np.ones((m,1)), muab), axis=1)
    messagelhfRatiosNew /= np.sum(messagelhfRatiosNew, axis=1).reshape((-1,1))
    messagelhfRatiosNew = messagelhfRatiosNew[:,0]
    
    return assocProbExisting, assocProbNew, messagelhfRatios, messagelhfRatiosNew
