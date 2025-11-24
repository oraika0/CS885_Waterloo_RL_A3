import collections
import numpy as np
import random
import torch

# Replay buffer
# TODO: Adjust this replay buffer to handle hidden states?
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N):
        self.buf = collections.deque(maxlen = N)
    
    # add: add a transition (s, a, r, s2, d)
    def add(self, s, a, r):
        self.buf.append((s, a, r))
    
    # sample: return minibatch of size n
    def sample(self, n):
        minibatch = random.sample(self.buf, n)
        
        return minibatch
        
        # for mb in minibatch:
        #     s, a, r, s2, d = mb
        #     S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]

        # if type(A[0]) == int:
        #     return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
        # elif type(A[0]) == float:
        #     return t.f(S), t.f(A), t.f(R), t.f(S2), t.i(D)
        # else:
        #     return t.f(S), torch.stack(A), t.f(R), t.f(S2), t.i(D)

