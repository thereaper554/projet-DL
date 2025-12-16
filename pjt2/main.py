import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================
# XOR DATASET
# =========================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

X = X.T  # shape (2,4)
y = y.T  # shape (1,4)

# =========================
# SIGMOID FUNCTION
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================
# INITIAL WEIGHTS
# =========================
np.random.seed(42)
W1 = np.random.randn(2, 2)
b1 = np.zeros((2, 1))
W2 = np.random.randn(1, 2)
b2 = np.zeros((1, 1))

# =========================
# HYPERPARAMETERS
# =========================
lr = 0.1
epochs = 10000
m = X.shape[1]

# =========================
# TRAINING LOOP + LOSS TRACKING
# =========================
loss_history = []

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(W1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    # Loss
    loss = -np.mean(y*np.log(a2 + 1e-9) + (1-y)*np.log(1-a2 + 1e-9))
    loss_history.append(loss)

    # Backprop
    dZ2 = a2 - y
    dW2 = np.dot(dZ2, a1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (a1 * (1 - a1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # Gradient descent
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# =========================
# STREAMLIT INTERFACE
# =========================
st.title("XOR Neural Network with Plots")

# Input selection
x1 = st.selectbox("Select x1", [0, 1])
x2 = st.selectbox("Select x2", [0, 1])

x_input = np.array([[x1], [x2]])
a1_input = sigmoid(np.dot(W1, x_input) + b1)
a2_input = sigmoid(np.dot(W2, a1_input) + b2)
prediction = int(a2_input >= 0.5)

st.write(f"Prediction: **{prediction}**")
st.write(f"Network output (sigmoid): {a2_input[0,0]:.4f}")

# --------------------------
# Plot 1: Loss curve
# --------------------------
st.subheader("Loss Curve")
fig, ax = plt.subplots()
ax.plot(loss_history)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss over Epochs")
st.pyplot(fig)

# --------------------------
# Plot 2: XOR Decision Boundary
# --------------------------
st.subheader("XOR Decision Boundary")

# create a grid
xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, 200),
                     np.linspace(-0.1, 1.1, 200))
grid = np.c_[xx.ravel(), yy.ravel()].T
a1_grid = sigmoid(np.dot(W1, grid) + b1)
a2_grid = sigmoid(np.dot(W2, a1_grid) + b2)
pred_grid = (a2_grid >= 0.5).reshape(xx.shape)

fig2, ax2 = plt.subplots()
ax2.contourf(xx, yy, pred_grid, alpha=0.3, cmap=plt.cm.Paired)
ax2.scatter(X[0, :], X[1, :], c=y[0], edgecolors='k', s=100, cmap=plt.cm.Paired)
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_title("XOR Decision Boundary")
st.pyplot(fig2)

# --------------------------
# Display full XOR table
# --------------------------
st.subheader("Full XOR Table")
for i in range(4):
    x = X[:, i].reshape(2, 1)
    a1_i = sigmoid(np.dot(W1, x) + b1)
    a2_i = sigmoid(np.dot(W2, a1_i) + b2)
    pred_i = int(a2_i >= 0.5)
    st.write(f"{x[0,0]} XOR {x[1,0]} = {pred_i} (sigmoid={a2_i[0,0]:.4f})")
