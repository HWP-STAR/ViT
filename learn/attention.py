import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def self_attention(X,W_Q,W_K,W_V):
    Q=X @ W_Q
    K= X@ W_K
    V= X @ W_V

    scores=Q@ K.T
    d_k=Q.shape[-1]
    scores=scores/np.sqrt(d_k)

    attention_weights=softmax(scores,axis=-1)

    output=attention_weights @ V

    return output,attention_weights

X = np.array([
    [1, 0, 0],   # 词 A
    [0, 1, 0],   # 词 B
    [0, 0, 1]    # 词 C
])

# 定义投影矩阵（为了简单，设置 d_k = 2, d_v = 2）
W_Q = np.array([[1, 0], [0, 1], [0, 0]])   # (3,2)
W_K = np.array([[1, 0], [0, 1], [0, 0]])   # (3,2)
W_V = np.array([[1, 0], [0, 1], [0, 0]])   # (3,2)

output, attn = self_attention(X, W_Q, W_K, W_V)

print(f'X:\n{X}')
print(f"attention_weights:\n{attn}")
print(f'output:\n{output}')
