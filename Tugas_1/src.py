import numpy as np
import os, csv

# ---------------------------------------------------------
# 1. Fungsi aktivasi
# ---------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(output):
    # turunan sigmoid dinyatakan dalam bentuk output (g(z)), bukan z
    return output * (1 - output)


# ---------------------------------------------------------
# 2. Forward pass untuk 1 sample
# ---------------------------------------------------------
def forward_single(x, bias, weights):
    z = bias + np.dot(weights, x)       # dot product
    output = sigmoid(z)                  # g(z)
    prediction = 1 if output >= 0.5 else 0  # Prediksi
    return z, output, prediction


# ---------------------------------------------------------
# 3. Forward pass untuk batch (evaluasi)
# ---------------------------------------------------------
def forward_batch(X, bias, weights):
    z = bias + X @ weights
    output = sigmoid(z)
    prediction = (output >= 0.5).astype(int)
    return z, output, prediction


# ---------------------------------------------------------
# 4. Error & Sum Square Error
# ---------------------------------------------------------
def compute_error(y_true, output):
    error = y_true - output          # target - output
    sse = error ** 2                 # Sum Square Error per sample
    return error, sse


# ---------------------------------------------------------
# 5. Training loop (ONLINE / per-sample update)
# ---------------------------------------------------------
def train_slp(X, y, bias, weights, learning_rate=0.1, epochs=5, verbose=True):
    history = []

    for epoch in range(1, epochs + 1):
        sse_list = []
        pred_list = []

        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            # -- forward pass untuk 1 sample --
            z, output, pred = forward_single(x_i, bias, weights)
            pred_list.append(pred)

            # -- hitung error & SSE --
            error = y_i - output       # target - output
            sse = error ** 2
            sse_list.append(sse)

            # -- hitung gradient (turunan SSE terhadap bobot) --
            # d(SSE)/d(w) = 2 * (output - target) * g'(z) * x
            #             = -2 * error * g'(z) * x
            sig_deriv = sigmoid_derivative(output)
            grad_bias = 2 * (output - y_i) * sig_deriv
            grad_weights = 2 * (output - y_i) * sig_deriv * x_i

            # -- update bobot (gradient descent) --
            bias = bias - learning_rate * grad_bias
            weights = weights - learning_rate * grad_weights

        mse = np.mean(sse_list)
        acc = np.mean(np.array(pred_list) == y)
        history.append(mse)

        if verbose:
            print(f"Epoch {epoch:3d} | MSE = {mse:.10f} | Akurasi = {acc:.4f}")

    return bias, weights, history


# ---------------------------------------------------------
# 6. Prediksi & evaluasi (dipakai untuk data validasi)
# ---------------------------------------------------------
def predict(X, bias, weights):
    _, output, prediction = forward_batch(X, bias, weights)
    return output, prediction


def evaluate(X, y, bias, weights):
    output, prediction = predict(X, bias, weights)
    error, sse = compute_error(y, output)
    return {
        "mse": sse.mean(),
        "accuracy": (prediction == y).mean(),
        "outputs": output,
        "predictions": prediction,
    }


# ---------------------------------------------------------
# 7. Load dataset & split train/val sesuai spreadsheet
# ---------------------------------------------------------
def load_csv(filepath):
    """Membaca CSV tanpa header, mengembalikan X (float) dan y (int)."""
    X_list, y_list = [], []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            features = [float(v) for v in row[:4]]
            label = 0 if row[4].strip() == "Iris-setosa" else 1
            X_list.append(features)
            y_list.append(label)
    return np.array(X_list), np.array(y_list)


def split_like_spreadsheet(X, y):
    train_idx = list(range(0, 40)) + list(range(50, 90))   # 80 sampel
    val_idx   = list(range(40, 50)) + list(range(90, 100))  # 20 sampel

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx])


if __name__ == "__main__":
    # --- Load dataset ---
    csv_path = os.path.join(os.path.dirname(__file__), "PMM-SLP_Widad.xlsx - Data.csv")
    X, y = load_csv(csv_path)
    print(f"Dataset dimuat: {X.shape[0]} sampel, {X.shape[1]} fitur")
    print(f"Distribusi kelas: setosa={int(np.sum(y==0))}, versicolor={int(np.sum(y==1))}\n")

    # --- Split sesuai spreadsheet (80 train, 20 val) ---
    X_train, y_train, X_val, y_val = split_like_spreadsheet(X, y)
    print(f"Train: {len(y_train)} sampel  |  Val: {len(y_val)} sampel\n")

    # --- Inisialisasi bobot sesuai spreadsheet ---
    bias = 0.5
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    print(f"Bobot awal  -> bias={bias}, weights={weights}\n")

    # --- Training (5 epoch, online update) ---
    print("=" * 55)
    print("TRAINING")
    print("=" * 55)
    bias, weights, history = train_slp(
        X_train, y_train, bias, weights,
        learning_rate=0.1, epochs=5, verbose=True
    )

    # --- Evaluasi Validasi per epoch ---
    print("\n" + "=" * 55)
    print("TRAINING + VALIDATION PER EPOCH")
    print("=" * 55)

    bias = 0.5
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    learning_rate = 0.1

    for epoch in range(1, 6):
        sse_list = []
        pred_list = []

        for i in range(len(X_train)):
            x_i = X_train[i]
            y_i = y_train[i]

            z, output, pred = forward_single(x_i, bias, weights)
            pred_list.append(pred)
            error = y_i - output
            sse = error ** 2
            sse_list.append(sse)

            sig_deriv = sigmoid_derivative(output)
            grad_bias = 2 * (output - y_i) * sig_deriv
            grad_weights = 2 * (output - y_i) * sig_deriv * x_i

            bias = bias - learning_rate * grad_bias
            weights = weights - learning_rate * grad_weights

        train_mse = np.mean(sse_list)
        train_acc = np.mean(np.array(pred_list) == y_train)

        # Evaluasi validasi dengan bobot akhir epoch ini
        hasil_val = evaluate(X_val, y_val, bias, weights)
        val_mse = hasil_val["mse"]
        val_acc = hasil_val["accuracy"]

        print(f"Epoch {epoch} | Train MSE = {train_mse:.10f} | Train Acc = {train_acc:.4f} | Val MSE = {val_mse:.10f} | Val Acc = {val_acc:.4f}")
