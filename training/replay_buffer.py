"""training/replay_buffer.py — Grid replay buffer"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Transition:
    state: np.ndarray; action: np.ndarray; reward: float
    next_state: np.ndarray; consequence: float; done: bool
    hidden: np.ndarray; next_hidden: np.ndarray
    priority: float = 1.0

class GridReplayBuffer:
    def __init__(self, action_dim, capacity=100_000, seed=42):
        self.capacity = capacity; self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self._buf: list = []; self._pos = 0

    def push(self, state, action, reward, next_state, consequence, done, hidden, next_hidden):
        t = Transition(
            np.asarray(state,np.float32), np.asarray(action,np.float32),
            float(reward), np.asarray(next_state,np.float32),
            float(consequence), bool(done),
            np.asarray(hidden,np.float32).flatten(),
            np.asarray(next_hidden,np.float32).flatten())
        if len(self._buf) < self.capacity: self._buf.append(t)
        else: self._buf[self._pos] = t
        self._pos = (self._pos+1) % self.capacity

    def sample(self, batch_size):
        if len(self._buf) < batch_size: return None
        idxs = self.rng.choice(len(self._buf), batch_size, replace=False)
        b    = [self._buf[i] for i in idxs]
        return {
            "states":       np.stack([x.state       for x in b]),
            "actions":      np.stack([x.action      for x in b]),
            "rewards":      np.array([x.reward      for x in b],np.float32),
            "next_states":  np.stack([x.next_state  for x in b]),
            "consequences": np.array([x.consequence for x in b],np.float32),
            "dones":        np.array([x.done        for x in b],np.float32),
            "hiddens":      np.stack([x.hidden      for x in b]),
            "next_hiddens": np.stack([x.next_hidden for x in b]),
            "weights":      np.ones(batch_size,np.float32),
        }

    def __len__(self): return len(self._buf)
