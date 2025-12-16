import streamlit as st
import os
# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Logic Gate Neural Network", layout="wide")

st.title("🧠 Logic Gate Neural Network (From Scratch)")
st.write("Logistic Regression trained using **gradient descent**, exactly like your lecture formulas.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")

gate_type = st.sidebar.selectbox("Choose Logic Gate", ["AND", "OR"])
epochs = st.sidebar.slider("Epochs", 1000, 30000, 20000, step=1000)
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)

train_button = st.sidebar.button("🚀 Train Model")

# ==========================
# Training
# ==========================
if train_button:
    X, Y = get_dataset(gate_type)

    W, b, losses, grad_norms = train_model(X, Y, epochs, alpha)

    # Save model
    model = {"W": W, "b": b, "gate": gate_type}
    joblib.dump(model, "model.pkl")

    st.success("Model trained and saved with joblib!")

    # ==========================
    # Display parameters
    # ==========================
    st.subheader("📌 Final Parameters")
    st.write("Weights:", W.ravel())
    st.write("Bias:", b)

    # ==========================
    # Results Table
    # ==========================
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    pred = np.round(A)

    st.subheader("📊 Results Table")

    table_data = []
    for i in range(len(X)):
        table_data.append([
            X[i, 0],
            X[i, 1],
            round(Z[i, 0], 4),
            round(A[i, 0], 4),
            int(pred[i, 0])
        ])

    st.table(
        {
            "x1": [row[0] for row in table_data],
            "x2": [row[1] for row in table_data],
            "z": [row[2] for row in table_data],
            "sigmoid(z)": [row[3] for row in table_data],
            "prediction": [row[4] for row in table_data],
        }
    )

    # ==========================
    # Plots
    # ==========================
    st.subheader("📉 Training Dynamics")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(losses)
        ax1.set_title("Loss vs Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Binary Cross-Entropy Loss")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(grad_norms)
        ax2.set_title("Gradient Norm vs Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("||∇W||")
        st.pyplot(fig2)

# ==========================
# Prediction Section
# ==========================
st.divider()
st.subheader("🔮 Make a Prediction")

if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
    W = model["W"]
    b = model["b"]

    x1 = st.selectbox("x1", [0, 1])
    x2 = st.selectbox("x2", [0, 1])

    if st.button("Predict"):
        X_input = np.array([[x1, x2]])
        z = np.dot(X_input, W) + b
        a = sigmoid(z)
        prediction = int(np.round(a)[0][0])

        st.write("z =", float(z))
        st.write("sigmoid(z) =", float(a))
        st.success(f"Prediction: {prediction}")
else:
    st.warning("Train the model first to enable predictions.")
