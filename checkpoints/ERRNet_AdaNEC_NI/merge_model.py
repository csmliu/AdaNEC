import torch
model_weights = torch.load('../ERRNet_AdaNEC_OF/final_release.pt')['icnn']

# We use the domain-level RTAW weights for network interpolation, which are averaged on a dataset.
weights = [0.108686739, 0.009344623, 0.881968638]    # real20
# weights = [0.217511492, 0.111057787, 0.671430722]    # wild
# weights = [0.101291473, 0.580948268, 0.317760259]    # postcard
# weights = [0.186080102, 0.355502779, 0.458417119]    # solid


ckpt = {'icnn': {}}
for k in model_weights.keys():
    if k.startswith('0'):
        k_ = k[2:]
        ckpt['icnn'][k_] = model_weights['0.'+k_] * weights[0] \
                         + model_weights['1.'+k_] * weights[1] \
                         + model_weights['2.'+k_] * weights[2]

torch.save(ckpt, 'final_release.pt')