import torch.nn as nn
import MinkowskiEngine as ME


def get_norm(norm_type, num_feats, bn_momentum=0.05, dimension=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(num_feats)
  elif norm_type == 'INBN':
    return nn.Sequential(
        ME.MinkowskiInstanceNorm(num_feats),
        ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum))
  else:
    raise ValueError(f'Type {norm_type}, not defined')


def get_nonlinearity(non_type):
  if non_type == 'ReLU':
    return ME.MinkowskiReLU()
  elif non_type == 'ELU':
    # return ME.MinkowskiInstanceNorm(num_feats, dimension=dimension)
    return ME.MinkowskiELU()
  else:
    raise ValueError(f'Type {non_type}, not defined')
