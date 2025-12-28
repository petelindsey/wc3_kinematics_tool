# wc3kin/wc3mdl/retarget.py
def build_retarget_map(src_seq, dst_seq):
    ratio = dst_seq.duration / src_seq.duration
    return lambda t: int(t * ratio)