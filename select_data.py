import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error


def select_data():
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data[:5000], mnist.target.astype(int)[:5000]
    return x, y

def fit_scaler_transform(X_train, X_test, scaler = None):
    if scaler is None:
        scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def pca_transform(X_train, X_test, n_components=50):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def pca_accuracy(X_train_pca, X_test_pca, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=43)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def experiment_pca_components(X_train_scaled, X_test_scaled, y_train, y_test, comps):
    results = []
    for k in comps:
        t0 = time.time()
        pca_k = PCA(n_components=k, random_state=42)
        Xtr_k = pca_k.fit_transform(X_train_scaled)
        Xte_k = pca_k.transform(X_test_scaled)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(Xtr_k, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(Xte_k)
        acc = accuracy_score(y_test, y_pred)

        # reconstruction MSE (on scaled space)
        Xte_recon = pca_k.inverse_transform(Xte_k)
        mse = mean_squared_error(X_test_scaled, Xte_recon)

        cum_var = pca_k.explained_variance_ratio_.sum()

        results.append({"n_components":k, "accuracy":acc, "train_time":train_time,
                        "cum_explained_var":cum_var, "recon_mse":mse})
        print(f"k={k}: acc={acc:.4f}, cum_var={cum_var:.4f}, mse={mse:.6f}")

        # save model & a small sample recon image
        import joblib, numpy as np
        joblib.dump(pca_k, f"artifacts/pca_k_{k}.joblib")
        np.save(f"artifacts/Xte_recon_k{k}.npy", Xte_recon[:100])  # keep sample

    df = pd.DataFrame(results)
    df.to_csv("artifacts/pca_experiment_results.csv", index=False)
    return df

def main():
    X, y = select_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled, X_test_scaled, scaler_obj = fit_scaler_transform(X_train, X_test, StandardScaler())

    X_train_pca, X_test_pca, pca = pca_transform(X_train_scaled, X_test_scaled, n_components=50)

    print("After scaling:")
    print("Accuracy on raw pixels:", pca_accuracy(X_train_scaled, X_test_scaled, y_train, y_test))
    print("Accuracy on PCA:", pca_accuracy(X_train_pca, X_test_pca, y_train, y_test))

    df_results = experiment_pca_components(X_train_scaled, X_test_scaled, y_train, y_test, comps=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(df_results)

    return X, y

 
if __name__ == "__main__":
    X, y = main()

# # Feature importance on raw pixels
# clf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_raw.fit(X_train_scaled, y_train)
# y_pred_raw = clf_raw.predict(X_test_scaled)
# acc_raw = accuracy_score(y_test, y_pred_raw)
# print("Accuracy on raw pixels:", acc_raw)

# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_pca, y_train)

# y_pred = clf.predict(X_test_pca)
# accuracy = accuracy_score(y_test, y_pred)

# joblib.dump(pca, "pca_50.joblib")
# joblib.dump(clf, "rf_pca_50.joblib")
# joblib.dump(scaler, "scaler.joblib")

# print("Original shape:", X_train.shape)
# print("Shape after PCA:", X_train_pca.shape)
# print(f"Accuracy: {accuracy:.4f}")

# np.save("mnist_pca_components.npy", pca.components_)
# np.save("mnist_pca_explained_var.npy", pca.explained_variance_ratio_)

# plt.figure(figsize=(10, 6))  
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.title('Cumulative Explained Variance Ratio', fontsize=14)
# plt.xlabel('Number of Components', fontsize=12)
# plt.ylabel('Cumulative Explained Variance', fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# X_pca_10 = X_train_pca[:10]
# X_reconstructed = pca.inverse_transform(X_pca_10)

# fig, axes = plt.subplots(2, 10, figsize=(15, 4))
# for i in range(10):
#     axes[0, i].imshow(X_train.iloc[i].values.reshape(28, 28), cmap='gray')
#     axes[0, i].axis('off')

#     axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
#     axes[1, i].axis('off')

# axes[0, 0].set_ylabel('Original', fontsize=12)
# axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
# plt.suptitle('Comparison of First 10 MNIST Images Before and After PCA', fontsize=14)
# plt.tight_layout()
# plt.show()


























# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("Số chiều")
# plt.ylabel("Tỷ lệ phương sai tích lũy")
# plt.title("Explained variance của PCA trên MNIST")
# plt.grid()
# plt.show()

# # Chọn 10 ảnh đầu tiên
# X_pca_10 = X_train_pca[:10]
# X_reconstructed = pca.inverse_transform(X_pca_10)

# # Vẽ so sánh
# fig, axes = plt.subplots(2, 10, figsize=(15,3))
# for i in range(10):
#     # Ảnh gốc
#     axes[0, i].imshow(X_train.iloc[i].values.reshape(28,28), cmap="gray")
#     axes[0, i].axis("off")
#     # Ảnh sau PCA
#     axes[1, i].imshow(X_reconstructed[i].reshape(28,28), cmap="gray")
#     axes[1, i].axis("off")

# axes[0,0].set_ylabel("Gốc", fontsize=12)
# axes[1,0].set_ylabel("PCA-50", fontsize=12)
# plt.suptitle("So sánh ảnh MNIST trước và sau PCA (50 chiều)", fontsize=14)
# plt.show()