#%%
class A():
    def __init__(self):
        self.x = 0

    def __add__(self,number):
        self.x+=number

    def inc(self):
        self+=1

a=A()
a.inc()
a.inc()
a.x
# %%
print(1 is int)
# %%
a = []
print(isinstance(a, list))
# %%
import numpy as np
a = np.empty((1,3))
s = ['1','2','3','4']
a[:] = s[1:]
a

# %%
