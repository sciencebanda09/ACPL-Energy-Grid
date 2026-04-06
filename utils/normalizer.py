"""utils/normalizer.py"""
import numpy as np

class RunningNormalizer:
    def __init__(self, dim, eps=1e-8, clip=5.0):
        self.mean = np.zeros(dim,np.float64); self.var = np.ones(dim,np.float64)
        self.count = 0; self.eps = eps; self.clip = clip

    def update(self, x):
        x = np.asarray(x,np.float64)
        if x.ndim==1: x=x[None]
        n = x.shape[0]; bm = x.mean(0); bv = x.var(0)
        tot = self.count+n; d = bm-self.mean
        self.mean  = self.mean + d*n/tot
        self.var   = (self.var*self.count + bv*n + d**2*self.count*n/tot)/tot
        self.count = tot

    def normalize(self, x):
        x = np.asarray(x,np.float32)
        n = (x-self.mean.astype(np.float32))/(np.sqrt(self.var.astype(np.float32))+self.eps)
        return np.clip(n,-self.clip,self.clip).astype(np.float32)
