"""CPU reference arithmetic: attention visibility, toy routing, exact format byte counts.
Not V4 attention, no RoPE/MLA/compressor/sink/HC, no real FP8/FP4 arithmetic or kernels.
"""
import math
import json

def attention(query, keys, values, visible):
    if not visible:
        raise ValueError('empty reference visibility')
    if len(keys) != len(values) or not query:
        raise ValueError('shape mismatch')
    if any(len(k) != len(query) for k in keys):
        raise ValueError('Q/K dimension mismatch')
    if any(len(v) != len(values[0]) for v in values):
        raise ValueError('V dimension mismatch')
    scores=[sum(a*b for a,b in zip(query,keys[i]))/math.sqrt(len(query)) for i in visible]
    peak=max(scores);weights=[math.exp(s-peak) for s in scores];denom=sum(weights)
    probs=[w/denom for w in weights]
    return [sum(p*values[i][d] for p,i in zip(probs,visible)) for d in range(len(values[0]))]

def paged_view(blocks, block_table, length):
    flat=[token for physical in block_table for token in blocks[physical]]
    if length > len(flat) or length < 0:
        raise ValueError('block table cannot cover length')
    return flat[:length]

def route(scores,bias,topk,logical_to_physical):
    """Toy: correction changes selection; raw positive scores define normalized weights.
    This is NOT a reimplementation of tile_kernels sqrtsoftplus/noaux_tc.
    """
    if len(scores)!=len(bias) or not 1<=topk<=len(scores) or any(x<0 for x in scores):
        raise ValueError('invalid toy gate inputs')
    chosen=sorted(range(len(scores)),key=lambda i:(-(scores[i]+bias[i]),i))[:topk]
    total=sum(scores[i] for i in chosen)
    if total<=0:raise ValueError('zero route weight')
    return [(i,logical_to_physical[i][0],scores[i]/total) for i in chosen]

def combine(x,routes,expert_outputs):
    """Fixed outputs stand in for grouped GEMM/SwiGLU; x is a toy shared path."""
    return [x[d]+sum(weight*expert_outputs[logical][d] for logical,_,weight in routes) for d in range(len(x))]

def kv_bytes(elements,bits,per_scale,scale_bytes):
    if elements<1 or bits not in (4,8) or per_scale<1 or scale_bytes<1 or elements*bits%8:
        raise ValueError('invalid packed format')
    return elements*bits//8 + math.ceil(elements/per_scale)*scale_bytes

if __name__=='__main__':
    keys=[[1.,0.],[0.,1.],[1.,1.],[9.,9.]]
    values=[[1.,0.],[0.,1.],[2.,2.],[99.,99.]]
    # Physical block 1 contains logical positions 0..1, physical 0 contains 2..3.
    stored=[values[2:],values[:2]]
    out=attention([1.,0.],keys,paged_view(stored,[1,0],4),[0,1])
    routes=route([.6,.3,.1],[0.,0.,.5],2,{0:[3],1:[1],2:[0,2]})
    print(json.dumps({'reference_attention':out,'logical_physical_weight':routes,
      'toy_moe_output':combine(out,routes,{0:[2.,0.],1:[0.,2.],2:[1.,1.]}),
      'format_bytes_per_token':{'V41_MXFP8':kv_bytes(512,8,32,1),'V41_FP4':kv_bytes(512,4,16,1),'FP4_INDEX':kv_bytes(128,4,32,1)},
      'scope':'reference arithmetic and layout accounting only'},indent=2))
