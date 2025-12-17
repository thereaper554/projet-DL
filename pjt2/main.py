import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# STREAMLIT HEADER 
# ==============================
st.set_page_config(page_title="XOR Neural Network", layout="wide")
st.title("XOR Neural Network")

# ==============================
# SIDEBAR CONTROLS 
# ==============================
lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 1000, 20000, 10000, 1000)

# ==============================
# XOR DATASET 
# ==============================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

X = X.T   # shape (2,4)
y = y.T   # shape (1,4)

# ==============================
# SIGMOID ACTIVATION FUNCTION
# ==============================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ==============================
# INITIAL WEIGHTS 
# ==============================
W1 = np.random.randn(2, 2) * 1
b1 = np.zeros((2, 1))

W2 = np.random.randn(1, 2) * 1
b2 = np.zeros((1, 1))

m = X.shape[1]

losses = []   # (NEW) only for plotting

# ==============================
# TRAINING 
# ==============================
for epoch in range(epochs):

    # FORWARD PASS
    z1 = np.dot(W1, X) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    # LOSS
    loss = -np.mean(y * np.log(a2 + 1e-9) +
                    (1 - y) * np.log(1 - a2 + 1e-9))
    losses.append(loss)  

    # BACKPROP
    dZ2 = a2 - y
    dW2 = np.dot(dZ2, a1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (a1 * (1 - a1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # GRADIENT DESCENT
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# ==============================
# VISUALIZATION  
# ==============================
col1, col2 = st.columns(2)

# ---- LOSS CURVE ----
with col1:
    st.subheader("Training Loss")
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    st.pyplot(fig)

# ---- DECISION BOUNDARY ----
with col2:
    st.subheader("Decision Boundary")

    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200),
                         np.linspace(-0.2, 1.2, 200))

    grid = np.c_[xx.ravel(), yy.ravel()].T

    z1 = np.dot(W1, grid) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    preds = sigmoid(z2).reshape(xx.shape)

    fig2, ax2 = plt.subplots()
    ax2.contourf(xx, yy, preds, levels=20, alpha=0.7)
    ax2.scatter(X[0], X[1], c=y.flatten(), edgecolors="k", s=100)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    st.pyplot(fig2)

# ==============================
# FINAL RESULTS TABLE  
# ==============================
st.subheader("Final XOR Results")

rows = []
for i in range(4):
    x = X[:, i].reshape(2, 1)

    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    rows.append([
        int(x[0,0]),
        int(x[1,0]),
        round(z1[0,0], 3),
        round(z1[1,0], 3),
        round(a1[0,0], 3),
        round(a1[1,0], 3),
        round(a2[0,0], 3),
        int(a2 >= 0.5)
    ])

st.table({
    "x1": [r[0] for r in rows],
    "x2": [r[1] for r in rows],
    "z1[0]": [r[2] for r in rows],
    "z1[1]": [r[3] for r in rows],
    "a1[0]": [r[4] for r in rows],
    "a1[1]": [r[5] for r in rows],
    "sigmoid(z2)": [r[6] for r in rows],
    "Prediction": [r[7] for r in rows],
})


