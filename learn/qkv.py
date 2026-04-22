import numpy as np 

# 位置1: "我"  位置2: "爱"  位置3: "你"
encoder_outputs = np.array([
    [1.0, 0.0, 0.5, 0.2],   # "我" 的表示
    [0.0, 1.0, 0.3, 0.1],   # "爱" 的表示
    [0.5, 0.5, 0.8, 0.4]    # "你" 的表示
])  # shape (3, 4)

d_k=2 #dim of Q,K
d_v=3


W_Q = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.1]])  # (4,2)
W_K = np.array([[1, 0], [0, 1], [0.3, 0.2], [0.1, 0.2]])  # (4,2)
W_V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]])  # (4,3)

Q=encoder_outputs @ W_Q
K=encoder_outputs @ W_K
V=encoder_outputs @ W_V

print("Query 矩阵 Q (每个位置一个查询向量):")
print(Q)
print("\nKey 矩阵 K (每个位置一个键向量):")
print(K)
print("\nValue 矩阵 V (每个位置的值向量):")
print(V)

q_dec = np.array([[0.8, 0.2]])   # 解码器的初始查询

# 1. 计算相似度 (点积)
scores = q_dec @ K.T   # (1, 3)
print("相似度得分 (与'我','爱','你'):", scores[0])

# 2. 缩放
d_k = K.shape[-1]   # 2
scores = scores / np.sqrt(d_k)   # sqrt(2)=1.414
print("缩放后:", scores[0])

# 3. Softmax 得到注意力权重
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

weights = softmax(scores[0])
print("注意力权重 (三个位置之和=1):", weights)

# 4. 加权求和 Value
context = weights.reshape(1, -1) @ V   # (1, 3)
print("最终的上下文向量:", context[0])
