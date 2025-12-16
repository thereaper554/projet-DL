import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(page_title="OR Gate Perceptron", layout="centered")

st.title("Single Perceptron – OR Logic Gate")
st.markdown(
    """
    This application demonstrates **logistic regression (single perceptron)**  
    trained from scratch on the **OR logic gate** using **exact lecture formulas**.
    """
)

# =========================================================
# OR Gate Dataset
# =========================================================
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([[0], [1], [1], [1]])
n = X.shape[0]

# =========================================================
# Sidebar – Hyperparameters
# =========================================================
st.sidebar.header("Training Parameters")

alpha = st.sidebar.slider("Learning rate (α)", 0.01, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 1000, 30000, 20000, step=1000)

# =========================================================
# Sigmoid (lecture formula)
# =========================================================
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# =========================================================
# Training function
# =========================================================
def train_model(X, Y, alpha, epochs):
    np.random.seed(1)
    W = np.random.randn(2, 1) * 0.1
    b = 0.0

    losses = []
    grad_norms = []

    for _ in range(epochs):
        Z = np.dot(X, W) + b
        A = sigmoid(Z)

        # Binary Cross Entropy
        L = -(1/n) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
        losses.append(L)

        dW = (1/n) * np.dot(X.T, (A - Y))
        db = (1/n) * np.sum(A - Y)

        grad_norms.append(np.linalg.norm(dW))

        W = W - alpha * dW
        b = b - alpha * db

    return W, b, losses, grad_norms

# =========================================================
# Train button
# =========================================================
if st.button("Train Perceptron"):
    W, b, losses, grad_norms = train_model(X, Y, alpha, epochs)

    # Save model
    joblib.dump({"W": W, "b": b}, "model.joblib")

    st.success("Model trained and saved successfully.")

    # =====================================================
    # Loss plot
    # =====================================================
    st.subheader("Loss Minimization")

    fig1, ax1 = plt.subplots()
    ax1.plot(losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Binary Cross Entropy Loss")
    st.pyplot(fig1)

    st.write(f"**Minimum loss achieved:** `{min(losses):.6f}`")

    # =====================================================
    # Gradient norm plot
    # =====================================================
    st.subheader("Gradient Evolution")

    fig2, ax2 = plt.subplots()
    ax2.plot(grad_norms)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("||∇W||")
    ax2.set_title("Gradient Norm Over Training")
    st.pyplot(fig2)

    # =====================================================
    # Final predictions table
    # =====================================================
    st.subheader("Final Predictions (OR Gate)")

    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    preds = np.round(A)

    st.table({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "z": Z.ravel(),
        "sigmoid(z)": A.ravel(),
        "prediction": preds.ravel().astype(int)
    })

# =========================================================
# Load existing model
# =========================================================
if os.path.exists("model.joblib"):
    if st.button("Load Saved Model"):
        model = joblib.load("model.joblib")
        W, b = model["W"], model["b"]

        st.success("Saved model loaded.")

        Z = np.dot(X, W) + b
        A = sigmoid(Z)
        preds = np.round(A)

        st.table({
            "x1": X[:, 0],
            "x2": X[:, 1],
            "z": Z.ravel(),
            "sigmoid(z)": A.ravel(),
            "prediction": preds.ravel().astype(int)
        })
