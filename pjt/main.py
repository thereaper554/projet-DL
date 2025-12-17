import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Perceptron – Logic Gates",
    layout="centered"
)

st.title("Single Perceptron for AND / OR Logic Gates")


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Configuration")

gate = st.sidebar.selectbox(
    "Logic Gate",
    ["AND", "OR"]
)

alpha = st.sidebar.slider(
    "Learning rate (α)",
    min_value=0.01,
    max_value=1.0,
    value=0.1
)

epochs = st.sidebar.slider(
    "Epochs",
    min_value=1000,
    max_value=30000,
    value=20000,
    step=1000
)

# =========================================================
# Dataset definition
# =========================================================
def get_dataset(gate):
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    if gate == "AND":
        Y = np.array([[0], [0], [0], [1]])
    else:  
        Y = np.array([[0], [1], [1], [1]])

    return X, Y

X, Y = get_dataset(gate)
n = X.shape[0]

# =========================================================
# Sigmoid function  
# =========================================================
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# =========================================================
# Training f
# =========================================================
def train_perceptron(X, Y, alpha, epochs):
    np.random.seed(1)
    W = np.random.randn(2, 1) * 0.1
    b = 0.0

    losses = []
    grad_norms = []

    for _ in range(epochs):
        # Forward pass
        Z = np.dot(X, W) + b
        A = sigmoid(Z)

        # Binary Cross Entropy
        L = -(1/n) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        losses.append(L)

        # Gradients
        dW = (1/n) * np.dot(X.T, (A - Y))
        db = (1/n) * np.sum(A - Y)

        grad_norms.append(np.linalg.norm(dW))

        # Update
        W = W - alpha * dW
        b = b - alpha * db

    return W, b, losses, grad_norms

# =========================================================
# Train button
# =========================================================
if st.button("Train Perceptron"):
    W, b, losses, grad_norms = train_perceptron(X, Y, alpha, epochs)

    # Save model  
    model_name = f"model_{gate.lower()}.pkl"
    joblib.dump({"W": W, "b": b}, model_name)

    st.success("Training completed successfully.")

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
    # Gradient evolution plot
    # =====================================================
    st.subheader("Gradient Evolution")

    fig2, ax2 = plt.subplots()
    ax2.plot(grad_norms)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("||∇W||")
    ax2.set_title("Gradient Norm Over Training")
    st.pyplot(fig2)
        
    # ==============================
    # Decision Boundary pour AND/OR
    # ==============================
    st.subheader("Decision Boundary")
    
    # Crée une grille de points dans le plan x1-x2
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200),
                         np.linspace(-0.2, 1.2, 200))
    
    # Transforme la grille en colonnes pour le perceptron
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Calcul du forward pass pour chaque point de la grille
    Z = np.dot(grid, W) + b
    A = 1 / (1 + np.exp(-Z))  # sigmoid
    
    # Arrondir pour obtenir les classes 0 ou 1
    preds = np.round(A).reshape(xx.shape)
    
    # Affichage
    fig3, ax3 = plt.subplots()
    ax3.contourf(xx, yy, preds, levels=[-0.1, 0.5, 1.1], alpha=0.5, cmap="coolwarm")
    ax3.scatter(X[:, 0], X[:, 1], c=Y.ravel().astype(float), edgecolors="k", s=100, cmap="coolwarm")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_title(f"Decision Boundary pour {gate} Gate")
    st.pyplot(fig3)


    # =====================================================
    # Final results table
    # =====================================================
    st.subheader(f"Final Predictions ({gate} Gate)")

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
