from types import MethodType

import torch

def register_forward(model, model_name):
    if model_name.split('_')[0] == 'deit':
        model.forward = MethodType(deit_forward, model)
    elif model_name.split('_')[0] == 'cait':
        model.forward = MethodType(cait_forward, model)
    elif model_name.split('_')[0] == 'swin':
        model.forward = MethodType(swin_forward, model)
    else:
        raise RuntimeError(f'Not defined customized method forward for model {model_name}')

def cait_forward(self, x, indices=None, require_feat: bool = True):
    if require_feat:
        x, block_outs = self.forward_intermediates(x, indices)
        x = self.forward_head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def deit_forward(self, x, indices=None, require_feat: bool = True):
    if require_feat:
        x, block_outs = self.forward_intermediates(x, indices)
        x = self.forward_head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def swin_forward(self, x, indices=None, require_feat: bool = True):
    if require_feat:
        x, block_outs = self.forward_intermediates(x, indices)
        x = self.forward_head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


