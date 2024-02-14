# %% [markdown]
# # 1. Understanding Attention
#
# - Before running the jupyter notebook, don't forget to copy it into your drive **(`File` => `Save a copy in Drive`)**. *Failing to do this step may result in losing the progress of your code.*
# - For this notebook, please fill in the line(s) directly after a `#TODO` comment with your answers.
# - For the submission of the assignment, please download this notebook as a **Python file**, named `A2S1.py`.

# %% [markdown]
# ## Imports and Setup

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# %%
torch.manual_seed(447)

key = torch.randn(4, 3)
key /= torch.norm(key, dim=1, keepdim=True)
key.round_(decimals=2)

value = torch.randn(4, 3)
value /= torch.norm(value, dim=1, keepdim=True)
value.round_(decimals=2)

print(f'key:\n{key}')
print(f'value:\n{value}')

# %%


def attention(query, key, value):
    """
    Note that we remove scaling for simplicity.
    """
    return F.scaled_dot_product_attention(query, key, value, scale=1)


def check_query(query, target, key=key, value=value, output=False):
    """
    Helper function for you to check if your query is close to the required target matrix.
    """
    a_out = attention(query, key, value)
    if output:
        print(f"attention = {a_out}")
        print(f"target = {target}")
        print("maximum absolute element-wise difference:",
              (target - a_out).abs().max())
    return (target - a_out).abs().max()

# %% [markdown]
# ## 1.2. Selection via Attention

# %%
# Define a query vector to ”select” the first value vector

# We want to find a scaler, c, such that
# exp(c) out weights 1 by a lot. Thus,
# the exp(c) / (exp(c) + 3) would be close to 1


out = torch.zeros((1, 4))
out[0, 0] = 100
query121 = torch.linalg.lstsq(key, out.T)[0].T
print(query121)
check_query(query121, value[0], output=True)

# %%
# Define a query matrix which results in an identity mapping – select all the value vectors

# This is the same rationale as previous one
# Now we just need an identity function instead
out = torch.eye(4, 4) * 100
q = torch.linalg.lstsq(key, out.T)
query122 = q[0].T
# compare output of attention with desired output
print(query122)
check_query(query122, value, output=True)

# %% [markdown]
# ## 1.3. Averaging via Attention

# %%
# define a query vector which averages all the value vectors

# The goal is that after softmax, the resulting output
# should be equally weighted (1 / 4). And we know
# exp(0) is just 1
out = torch.zeros((1, 4))
q = torch.linalg.lstsq(key, out.T)
query131 = q[0].T
# compare output of attention with desired output
print(query131)
target = torch.reshape(value.mean(0, keepdims=True),
                       (3,))  # reshape to a vector
check_query(query131, target, output=True)

# %%
# define a query vector which averages the first two value vectors
# We want to out to be equally weighted (1 / 2) for the first two entries
# so we make sure to scale up to override the 1's from the last two entries

# In addition, we also want to ensure that
# query can be solved out properly by considering
# the components of key
out = torch.zeros((1, 4))
out[0, 0] = 3
out[0, 1] = 3
out[0, 2] = -2
out[0, 3] = -3
print(f"Expected attn before softmax = {out}")
print(f"Expected attn after softmax = {torch.softmax(out, dim=-1)}")
query132 = torch.linalg.lstsq(key, out.T)[0].T
print(f"Actual attn before softmax = {query132 @ key.T}")
print(f"Actual attn after softmax = {torch.softmax(query132 @ key.T, dim=-1)}")
# compare output of attention with desired output
print(query132)

target = torch.reshape(value[(0, 1),].mean(
    0, keepdims=True), (3,))  # reshape to a vector
check_query(query132, target, output=True)

# %% [markdown]
# ## 1.4. Interactions within Attention

# %%
# Define a replacement for only the third key vector k[2] such that the result of attention
# with the same unchanged query q from (1.3.2) averages the first three value vectors.
m_key = key.clone()

# TODO:
m_key[2] = key[0]

# compare output of attention with desired output
check_query(query132, value[(0, 1, 2),].mean(0, keepdims=True), key=m_key)

# %%
# Define a replacement for only the third key vector k[2] such that the result of attention
# with the same unchanged query q from (1.3.2) returns the third value vector v[2].
m_key = key.clone()

# TODO:
# A = Q K.T
out = torch.zeros((4, 1))
out[:2, 0] = -3
out[2, 0] = 1
out[3:, 0] = -3
m_key = torch.linalg.lstsq(query132, out.T)[0].T
m_key[2] /= m_key[2].norm()
print(query132 @ m_key.T)
print(torch.softmax(query132 @ m_key.T, dim=-1))
# compare output of attention with desired output
check_query(query132, value[2], key=m_key)
