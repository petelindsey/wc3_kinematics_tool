# wc3kin/wc3mdl/reduce_keys.py
import numpy as np

def reduce_keys(track, eps=1e-4):
    times = sorted(track)
    if len(times)<=2:
        return track

    reduced = {times[0]:track[times[0]]}

    for i in range(1,len(times)-1):
        t0,t1,t2 = times[i-1],times[i],times[i+1]
        p0,p1,p2 = track[t0],track[t1],track[t2]
        interp = p0 + (p2-p0)*((t1-t0)/(t2-t0))
        if np.linalg.norm(interp-p1) > eps:
            reduced[t1]=p1

    reduced[times[-1]]=track[times[-1]]
    return reduced