import numpy as np
from scipy.signal import correlate
from scipy.optimize import least_squares
SPEED_OF_SOUND=343.0

def estimate_delay(a,b,fs,max_delay=0.08):
    a=np.asarray(a,float);b=np.asarray(b,float);n=min(len(a),len(b))
    if n<256:return 0.0
    a=a[-n:]-np.mean(a[-n:]);b=b[-n:]-np.mean(b[-n:])
    if np.std(a)>1e-12:a/=np.std(a)
    if np.std(b)>1e-12:b/=np.std(b)
    c=correlate(a,b,mode="full",method="fft");lags=np.arange(-n+1,n)
    mask=np.abs(lags)<=int(max_delay*fs)
    return float(lags[mask][np.argmax(np.abs(c[mask]))]/fs)

def multilateration(positions,delays,reference):
    ids=[i for i in positions if i in delays]
    if len(ids)<3:raise ValueError("At least 3 phones are required")
    ref=np.asarray(positions[reference],float)
    pts=np.asarray([positions[i] for i in ids],float)
    others=[i for i in ids if i!=reference]
    def fun(s):
        dr=np.linalg.norm(s-ref)
        return [((np.linalg.norm(s-np.asarray(positions[i],float))-dr)/SPEED_OF_SOUND-delays[i])*SPEED_OF_SOUND for i in others]
    out=least_squares(fun,np.mean(pts,axis=0))
    r=np.asarray(fun(out.x))
    return {"position":out.x.astype(float).tolist(),"error":float(np.sqrt(np.mean(r*r))),"success":bool(out.success)}
