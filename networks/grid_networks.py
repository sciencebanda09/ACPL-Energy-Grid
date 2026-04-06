"""
Grid Networks — grid_networks.py
All neural network primitives for the energy grid project.
Self-contained — no dependency on the original ACPL networks.py.
"""
import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────

def relu(x):     return np.maximum(0.0, x)
def sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def tanh(x):     return np.tanh(np.clip(x, -20, 20))
def softplus(x):
    x = np.clip(x, -20, 20)
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
def softmax(x):
    e = np.exp(np.clip(x, -30, 30) - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

def d_relu(x):     return (x > 0).astype(np.float32)
def d_sigmoid(s):  return s * (1.0 - s)
def d_tanh(t):     return 1.0 - t**2
def d_softplus(x): return sigmoid(x)


# ── Linear + LayerNorm + MLP ──────────────────────────────────────────────────

class Linear:
    def __init__(self, in_dim, out_dim, rng, scale=None):
        scale = scale or np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, np.float32)
        self._last_x = None

    def forward(self, x):
        self._last_x = np.asarray(x, np.float32)
        return self._last_x @ self.W + self.b

    def backward(self, d_out):
        dW = self._last_x.T @ d_out
        db = d_out.sum(0)
        dx = d_out @ self.W.T
        return dx, dW, db

    def params(self): return [self.W, self.b]


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.g = np.ones(dim, np.float32)
        self.b = np.zeros(dim, np.float32)
        self.eps = eps; self._cache = None

    def forward(self, x):
        mu = x.mean(-1, keepdims=True); std = x.std(-1, keepdims=True) + self.eps
        x_hat = (x - mu) / std; self._cache = (x_hat, std)
        return self.g * x_hat + self.b

    def backward(self, d_out):
        x_hat, std = self._cache; B, D = d_out.shape[0], d_out.shape[-1]
        dg = (d_out * x_hat).sum(0); db_g = d_out.sum(0)
        dx_hat = d_out * self.g
        dx = (1.0 / (B * std)) * (B * dx_hat - dx_hat.sum(-1, keepdims=True)
             - x_hat * (dx_hat * x_hat).sum(-1, keepdims=True))
        return dx, dg, db_g

    def params(self): return [self.g, self.b]


class MLP:
    def __init__(self, dims, rng, scale=None):
        self.layers, self.norms = [], []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1], rng, scale))
            if i < len(dims) - 2: self.norms.append(LayerNorm(dims[i+1]))
        self._pre_acts = []

    def forward(self, x):
        x = np.asarray(x, np.float32); self._pre_acts = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.norms):
                x = self.norms[i].forward(x); self._pre_acts.append(x.copy()); x = relu(x)
        return x

    def backward(self, d_out):
        grads = []; d = d_out; n_h = len(self.norms)
        for i in reversed(range(len(self.layers))):
            if i < n_h:
                d = d * d_relu(self._pre_acts[i]); d, dg, db_n = self.norms[i].backward(d)
                grads = [dg, db_n] + grads
            dx, dW, db = self.layers[i].backward(d); grads = [dW, db] + grads; d = dx
        return d, grads

    def all_params(self):
        p = []
        for i, layer in enumerate(self.layers):
            p.extend(layer.params())
            if i < len(self.norms): p.extend(self.norms[i].params())
        return p


# ── Adam ──────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip=3.0):
        self.params = list(params); self.lr = lr; self.beta1 = beta1
        self.beta2 = beta2; self.eps = eps; self.clip = clip
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]; self.t = 0

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = np.clip(g, -self.clip, self.clip)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


# ── GRU Cell ──────────────────────────────────────────────────────────────────

class GRUCell:
    def __init__(self, input_dim, hidden_dim, rng):
        s = np.sqrt(2.0 / (input_dim + hidden_dim))
        def W(): return rng.normal(0, s, (input_dim,  hidden_dim)).astype(np.float32)
        def U(): return rng.normal(0, s, (hidden_dim, hidden_dim)).astype(np.float32)
        def b(): return np.zeros(hidden_dim, np.float32)
        self.Wr,self.Ur,self.br = W(),U(),b()
        self.Wz,self.Uz,self.bz = W(),U(),b()
        self.Wn,self.Un,self.bn = W(),U(),b()
        self._cache = None

    def forward(self, x, h):
        x = np.asarray(x,np.float32); h = np.asarray(h,np.float32)
        r = sigmoid(x@self.Wr + h@self.Ur + self.br)
        z = sigmoid(x@self.Wz + h@self.Uz + self.bz)
        n = tanh(   x@self.Wn + (r*h)@self.Un + self.bn)
        h_new = (1-z)*n + z*h; self._cache = (x,h,r,z,n); return h_new

    def params(self):
        return [self.Wr,self.Ur,self.br,self.Wz,self.Uz,self.bz,self.Wn,self.Un,self.bn]

    def zero_state(self, batch=1):
        return np.zeros((batch, self.Wr.shape[1]), np.float32)


def gru_batch_forward(cell, X, H):
    X = np.asarray(X,np.float32); H = np.asarray(H,np.float32)
    r = sigmoid(X@cell.Wr + H@cell.Ur + cell.br)
    z = sigmoid(X@cell.Wz + H@cell.Uz + cell.bz)
    n = tanh(   X@cell.Wn + (r*H)@cell.Un + cell.bn)
    return (1-z)*n + z*H


# ── GRU Policy Network (discrete — for baseline comparison) ───────────────────

class GRUPolicyNet:
    def __init__(self, state_dim, action_dim, gru_dim=64, hidden_dim=128,
                 n_layers=2, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.gru_dim = gru_dim; self.action_dim = action_dim
        self.gru      = GRUCell(state_dim, gru_dim, rng)
        self.trunk    = MLP([gru_dim]+[hidden_dim]*n_layers, rng)
        self.val_head = Linear(hidden_dim, 1,          rng, 0.01)
        self.adv_head = Linear(hidden_dim, action_dim, rng, 0.01)
        all_p = self.gru.params()+self.trunk.all_params()+self.val_head.params()+self.adv_head.params()
        self.optim = Adam(all_p, lr=lr)

    def forward(self, state, h):
        state = np.asarray(state,np.float32)
        if state.ndim==1: state=state[None]
        h_new = self.gru.forward(state,h); feat = self.trunk.forward(h_new)
        v = self.val_head.forward(feat); a = self.adv_head.forward(feat)
        return v+a-a.mean(-1,keepdims=True), h_new

    def zero_state(self,b=1): return self.gru.zero_state(b)
    def _flat(self): return self.gru.params()+self.trunk.all_params()+self.val_head.params()+self.adv_head.params()
    def copy_weights_from(self,o): [t.__setitem__(slice(None),s) for s,t in zip(o._flat(),self._flat())]
    def soft_update_from(self,o,tau=0.005): [t.__setitem__(slice(None),tau*s+(1-tau)*t) for s,t in zip(o._flat(),self._flat())]


# ── Gaussian Actor (continuous) ───────────────────────────────────────────────

class GaussianActorGRU:
    LOG_STD_MIN = -4.0; LOG_STD_MAX = 1.5

    def __init__(self, state_dim, action_dim, gru_dim=64, hidden_dim=128,
                 n_layers=2, lr=8e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.action_dim = action_dim; self.gru_dim = gru_dim
        self.gru       = GRUCell(state_dim, gru_dim, rng)
        self.trunk     = MLP([gru_dim]+[hidden_dim]*n_layers, rng, scale=0.05)
        self.mean_head = Linear(hidden_dim, action_dim, rng, 0.01)
        self.lsd_head  = Linear(hidden_dim, action_dim, rng, 0.01)
        all_p = self.gru.params()+self.trunk.all_params()+self.mean_head.params()+self.lsd_head.params()
        self.optim = Adam(all_p, lr=lr)

    def zero_state(self, b=1): return self.gru.zero_state(b)

    def forward(self, s, h):
        h_new = self.gru.forward(s, h); feat = self.trunk.forward(h_new)
        mean  = self.mean_head.forward(feat)
        lsd   = np.clip(self.lsd_head.forward(feat), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean.squeeze(0), lsd.squeeze(0), h_new

    def sample(self, s, h, eval_mode=False):
        mean, lsd, h_new = self.forward(s, h)
        if eval_mode: return np.tanh(mean), 0.0, h_new, np.zeros_like(mean)
        std = np.exp(lsd); eps = np.random.standard_normal(self.action_dim).astype(np.float32)
        raw = mean + std * eps; action = np.tanh(raw)
        lp  = (-0.5*(eps**2+2*lsd+np.log(2*np.pi)) - np.log(1-action**2+1e-6)).sum()
        return action, float(lp), h_new, eps

    def _flat(self): return self.gru.params()+self.trunk.all_params()+self.mean_head.params()+self.lsd_head.params()


# ── Critic (continuous) ───────────────────────────────────────────────────────

class CriticGRU:
    def __init__(self, state_dim, gru_dim=64, hidden_dim=128, n_layers=2, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.gru   = GRUCell(state_dim, gru_dim, rng)
        self.trunk = MLP([gru_dim]+[hidden_dim]*n_layers, rng, scale=0.05)
        self.vhead = Linear(hidden_dim, 1, rng, 0.01)
        all_p = self.gru.params()+self.trunk.all_params()+self.vhead.params()
        self.optim = Adam(all_p, lr=lr)

    def forward(self, s, h):
        H = gru_batch_forward(self.gru, s, h)
        return self.vhead.forward(self.trunk.forward(H)).squeeze(-1)

    def zero_state(self, b=1): return self.gru.zero_state(b)

    def backward_update(self, s, h, targets, weights):
        H = gru_batch_forward(self.gru, s, h); feat = self.trunk.forward(H)
        v = self.vhead.forward(feat).squeeze(-1); B = len(targets)
        err = np.clip((targets-v)*weights, -5, 5)
        d_v, dWv, dbv = self.vhead.backward((err/B)[:,None])
        d_f, tg = self.trunk.backward(d_v)
        self.optim.step([np.zeros_like(p) for p in self.gru.params()] + tg + [dWv, dbv])
        return float(np.mean(err**2))

    def _flat(self): return self.gru.params()+self.trunk.all_params()+self.vhead.params()


# ── Multi-Horizon Consequence Net ─────────────────────────────────────────────

class MultiHorizonConsequenceNet:
    def __init__(self, state_dim, action_dim, hidden_dim=64, n_layers=2, lr=5e-4, seed=0):
        rng = np.random.default_rng(seed); self.action_dim = action_dim
        inp = state_dim + action_dim

        def _t(): return MLP([inp]+[hidden_dim]*n_layers, rng)
        def _h():
            return {hz: {"f": Linear(hidden_dim,1,rng,0.01),
                         "u": Linear(hidden_dim,1,rng,0.01),
                         "d": Linear(hidden_dim,1,rng,0.01)}
                    for hz in ("short","mid","long")}

        self.trunk_a,self.heads_a = _t(),_h()
        self.trunk_b,self.heads_b = _t(),_h()
        self.log_alpha = np.array([0.5],np.float32)
        self.log_beta  = np.array([0.3],np.float32)
        self.log_gamma = np.array([0.2],np.float32)
        self.horizon_w = np.zeros(3,np.float32)
        self.optim = Adam(self._all_params(), lr=lr)

    def _hp(self,h):
        p=[]
        for hz in ("short","mid","long"):
            for k in ("f","u","d"): p.extend(h[hz][k].params())
        return p

    def _all_params(self):
        return (self.trunk_a.all_params()+self.trunk_b.all_params()+
                self._hp(self.heads_a)+self._hp(self.heads_b)+
                [self.log_alpha,self.log_beta,self.log_gamma,self.horizon_w])

    @property
    def alpha(self): return float(softplus(self.log_alpha)[0])
    @property
    def beta(self):  return float(softplus(self.log_beta)[0])
    @property
    def gamma(self): return float(softplus(self.log_gamma)[0])
    @property
    def horizon_blend(self): return softmax(self.horizon_w[None]).squeeze()

    def _tf(self, trunk, heads, states, actions):
        oh = np.eye(max(self.action_dim,1),dtype=np.float32)[np.clip(actions.astype(int),0,self.action_dim-1)]
        x  = np.concatenate([states,oh],-1); H = trunk.forward(x)
        a,b,g = self.alpha, self.beta, self.gamma; out={}
        for key in ("short","mid","long"):
            F=softplus(heads[key]["f"].forward(H)); U=softplus(heads[key]["u"].forward(H))
            D=softplus(heads[key]["d"].forward(H))
            out[key]=(a*F+b*U+g*D).squeeze(-1)
        return out,H

    def forward(self, states, actions):
        ha,_=self._tf(self.trunk_a,self.heads_a,states,actions)
        hb,_=self._tf(self.trunk_b,self.heads_b,states,actions)
        w=self.horizon_blend
        Ca=w[0]*ha["short"]+w[1]*ha["mid"]+w[2]*ha["long"]
        Cb=w[0]*hb["short"]+w[1]*hb["mid"]+w[2]*hb["long"]
        return 0.5*(Ca+Cb), 0.5*(ha["short"]+hb["short"]), \
               0.5*(ha["mid"]+hb["mid"]), 0.5*(ha["long"]+hb["long"]), np.abs(Ca-Cb)

    def predict(self, states, actions):
        C,_,_,_,sigma = self.forward(states,actions); return C,sigma

    def update_step(self, states, actions, targets, weights):
        B = len(actions)
        oh = np.eye(max(self.action_dim,1),dtype=np.float32)[np.clip(actions.astype(int),0,self.action_dim-1)]
        x_in = np.concatenate([states,oh],-1); w=self.horizon_blend
        a,b,g = self.alpha,self.beta,self.gamma
        tgl=[]; hgl=[]; c_loss=0.0
        for trunk,heads in [(self.trunk_a,self.heads_a),(self.trunk_b,self.heads_b)]:
            H=trunk.forward(x_in); td=np.zeros_like(H); hg=[]
            for hz,hw in zip(("short","mid","long"),w):
                Fr=heads[hz]["f"].forward(H); Ur=heads[hz]["u"].forward(H); Dr=heads[hz]["d"].forward(H)
                F=softplus(Fr); U=softplus(Ur); D=softplus(Dr)
                C_h=(a*F+b*U+g*D).squeeze(-1)
                err=(C_h-targets)*weights/(B*2); c_loss+=float(np.mean(err**2))*float(hw)
                for hl,sc,raw in [(heads[hz]["f"],a,Fr),(heads[hz]["u"],b,Ur),(heads[hz]["d"],g,Dr)]:
                    ds=err[:,None]*sc*d_softplus(raw); dH,dW,db=hl.backward(ds); td+=dH; hg.extend([dW,db])
            _,tp=trunk.backward(td); tgl.append(tp); hgl.append(hg)
        C_t,_=self.predict(states,actions); be=(C_t-targets)*weights/B
        dw=be.mean()*np.ones(3,np.float32)
        d_a=np.array([be.mean()*a*0.01],np.float32)
        d_b=np.array([be.mean()*b*0.01],np.float32)
        d_g=np.array([be.mean()*g*0.01],np.float32)
        self.optim.step(tgl[0]+tgl[1]+hgl[0]+hgl[1]+[d_a,d_b,d_g,dw])
        return c_loss


# ── Lambda Net ────────────────────────────────────────────────────────────────

class LambdaNet:
    def __init__(self, state_dim=18, hidden_dim=32, lambda_max=4.0, lr=2e-4,
                 input_idx=None, seed=0):
        rng = np.random.default_rng(seed)
        self.lambda_max = lambda_max
        self.input_idx  = input_idx or [0, 3, 4]
        self.net   = MLP([len(self.input_idx),hidden_dim,hidden_dim,1], rng, scale=0.05)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def forward(self, state):
        s = np.asarray(state,np.float32); scalar = s.ndim==1
        if scalar: s=s[None]
        x   = s[:,self.input_idx]; raw = self.net.forward(x)
        out = sigmoid(raw).squeeze(-1)*self.lambda_max
        return float(out[0]) if scalar else out

    def backward_update(self, states, lam_errors, weight_decay=0.0):
        x   = np.asarray(states,np.float32)[:,self.input_idx]
        raw = self.net.forward(x); sig = sigmoid(raw).squeeze(-1)
        d_raw = (lam_errors*sig*(1-sig)*self.lambda_max)[:,None]
        _,grads = self.net.backward(d_raw)
        if weight_decay>0:
            params = self.net.all_params()
            grads  = [g+weight_decay*p for g,p in zip(grads,params)]
        self.optim.step(grads)

    def params(self): return self.net.all_params()


# ── Delay Estimator ───────────────────────────────────────────────────────────

class DelayEstimatorNet:
    def __init__(self, gru_dim=64, hidden_dim=32, tau_max=20, lr=2e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.tau_max   = tau_max
        self.n_classes = tau_max+1
        self.net   = MLP([gru_dim,hidden_dim,hidden_dim,self.n_classes], rng, scale=0.05)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def forward(self, h):
        h = np.asarray(h,np.float32); scalar = h.ndim==1
        if scalar: h=h[None]
        probs = softmax(self.net.forward(h))
        return probs[0] if scalar else probs

    def expected_delay(self, h):
        probs = self.forward(h); taus = np.arange(self.n_classes,dtype=np.float32)
        if probs.ndim==1: return float((probs*taus).sum())
        return (probs*taus[None,:]).sum(-1)

    def update(self, hiddens, observed_delays, weights=None):
        B = len(observed_delays); mask = observed_delays>=0
        if mask.sum()==0: return 0.0
        h_sel = np.asarray(hiddens,np.float32)[mask]
        tau_sel = observed_delays[mask].astype(int)
        w_sel = (weights[mask] if weights is not None else np.ones(mask.sum(),np.float32))
        logits = self.net.forward(h_sel); probs = softmax(logits)
        d_logits = probs.copy(); d_logits[np.arange(len(tau_sel)),tau_sel] -= 1.0
        d_logits *= (w_sel/B)[:,None]
        loss = float(-np.log(probs[np.arange(len(tau_sel)),tau_sel]+1e-8).mean())
        _,grads = self.net.backward(d_logits); self.optim.step(grads)
        return loss

    def params(self): return self.net.all_params()
