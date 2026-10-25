"""Simplified MTP budget objective. Synthetic costs/rates, not DSpark or timing."""
def choose(rates, curve, sampler_ms=0.0):
    if not rates or not curve or sampler_ms < 0:
        raise ValueError('nonempty requests/curve and nonnegative sampler cost required')
    if any(not 0 <= p <= 1 for seq in rates for p in seq):
        raise ValueError('confidence must be in [0,1]')
    if any(ms <= 0 for _, ms in curve) or any(a[0] >= b[0] for a,b in zip(curve,curve[1:])):
        raise ValueError('positive cost and increasing graph sizes required')
    queue = sorted(((p, i) for i, seq in enumerate(rates) for p in seq), key=lambda x:(-x[0],x[1]))
    expected = float(len(rates))
    best = None
    for extra in range(len(queue)+1):
        total = len(rates)+extra
        graph = next(((size,ms) for size,ms in curve if size >= total), None)
        if graph is not None:
            score = expected/(graph[1]+sampler_ms)
            if best is None or score > best[0]:
                best = score, extra, total, graph[0]
        if extra < len(queue):
            expected += queue[extra][0]
    if best is None:
        raise ValueError('no graph can hold golden tokens')
    lengths = [1]*len(rates)
    for _, i in queue[:best[1]]:
        lengths[i] += 1
    return {'query_lengths':lengths, 'total_q':best[2], 'graph_size':best[3], 'synthetic_score':round(best[0],4)}

if __name__ == '__main__':
    print(choose([[.9,.5],[.8,.1]], [(2,1.0),(4,1.3),(8,3.0)]))
