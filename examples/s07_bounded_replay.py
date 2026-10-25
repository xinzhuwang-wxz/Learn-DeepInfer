"""One-dimensional SWA visibility toy. Not complete YoCo/CP/cache equivalence."""
def visible(pos, data_len, window, keep, backtrack=0, valid_start=0):
    if window < 1 or backtrack < 0 or keep < window + backtrack:
        raise ValueError('keep must cover window + backtracking')
    if not 0 <= pos < data_len or valid_start < 0:
        raise ValueError('invalid absolute position')
    lo = min(pos+1, max(0, pos-window+1, data_len-keep, valid_start))
    return tuple(range(lo, pos+1))


def compare(length=12, window=4, keep=6, backtrack=2):
    full = {p: visible(p,length,window,keep,backtrack) for p in range(length)}
    compact = {p: visible(p,length,window,keep,backtrack) for p in range(max(0,length-keep),length)}
    assert all(full[p] == keys for p,keys in compact.items())
    return compact

if __name__ == '__main__':
    print(compare())
