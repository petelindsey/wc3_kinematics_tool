
# wc3kin/wc3mdl/interpolate.py
import numpy as np

def hermite(p0,p1,t0,t1,u):
    u2 = u*u
    u3 = u2*u
    h00 =  2*u3 - 3*u2 + 1
    h10 =      u3 - 2*u2 + u
    h01 = -2*u3 + 3*u2
    h11 =      u3 -    u2
    return h00*p0 + h10*t0 + h01*p1 + h11*t1

def sample_track(keys: dict[int, np.ndarray], t: int):
    times = sorted(keys)
    if t <= times[0]: return keys[times[0]]
    if t >= times[-1]: return keys[times[-1]]

    i = max(i for i in times if i <= t)
    j = min(j for j in times if j >= t)
    u = (t-i)/(j-i)

    p0 = keys[i]
    p1 = keys[j]
    return (1-u)*p0 + u*p1