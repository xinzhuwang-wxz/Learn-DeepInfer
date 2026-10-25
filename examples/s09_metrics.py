"""Arithmetic only: fixed-draft-length accounting. Synthetic inputs, no measurements."""
def decode_metrics(sps, requests, nextn, cp, acceptance=1.0, varlen=False):
    if varlen:
        raise ValueError('fixed-length acceptance rescaling is invalid for VARLEN=1')
    if sps <= 0 or requests < 1 or cp < 1 or nextn < 0 or not 0 <= acceptance <= 1:
        raise ValueError('invalid metric inputs')
    tau = 1 + nextn * acceptance
    return {'tokens_per_user_s': sps*tau, 'tokens_per_s_per_gpu':sps*requests*tau/cp, 'assumed_tokens_per_step':tau}

if __name__ == '__main__':
    print('full acceptance:', decode_metrics(20,64,5,1))
    print('assumed acceptance=.75:', decode_metrics(20,64,5,1,.75))
