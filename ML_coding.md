# Machine Learning Coding Interview Prep

以下内容按主题划分，每个 Topic 都包含了你在面试中可能被要求 **手写实现** 的核心问题。  
建议使用纯 Python / NumPy 实现，不依赖 sklearn / pytorch / tensorflow。

---

## 🧮 1. Core ML Algorithms from Scratch

### Topic: Linear & Logistic Regression
**Implement:**
- Linear Regression using Gradient Descent (with MSE loss)
- Logistic Regression (with cross-entropy + sigmoid, including numerical stability)
- Add L2 regularization (ridge)
- Implement `fit`, `predict`, and `score` methods

---

### Topic: KNN, Naive Bayes
**Implement:**
- K-Nearest Neighbors classifier (vectorized distance computation)
- Gaussian Naive Bayes (compute class priors, mean/var, and posterior)

---

### Topic: Decision Tree
**Implement:**
- ID3/CART style tree
- Gini and Entropy splitting
- Handle continuous features
- Add pruning or max-depth stopping rule

---

### Topic: K-Means Clustering
**Implement:**
- `kmeans(X, k, max_iter)`
- Random init (k-means++)
- Convergence check (distance threshold)
- Handle empty clusters

---

### Topic: PCA (Dimensionality Reduction)
**Implement:**
- `pca(X, k)` using SVD
- Project and reconstruct data
- Compute explained variance ratio

---

### Topic: Neural Network (2-layer)
**Implement:**
- Forward pass: `Linear -> ReLU -> Linear -> Softmax`
- Backpropagation manually (cross entropy)
- Gradient checking (numerical verification)

---

## 📊 2. Metrics & Model Evaluation

### Topic: Evaluation Metrics
**Implement:**
- Confusion matrix
- Precision, Recall, F1
- ROC curve and AUC (without sklearn)
- PR curve, average precision

---

### Topic: Cross Validation
**Implement:**
- `k_fold_cv(model, X, y, k)`
- Return mean ± std of metric
- Handle stratified sampling for classification

---

## 🧠 3. Vectorization & Data Manipulation

### Topic: NumPy Vectorization
**Implement:**
- Standardization `(X - mean)/std`
- Pairwise distance matrix (vectorized)
- Cosine similarity matrix
- Top-K selection using `np.argsort`
- Sliding window extraction function

---

### Topic: Pandas Data Processing
**Implement:**
- Groupby + aggregation tasks
- Merge/join on multiple keys
- Rolling window mean/median
- Missing value imputation
- Encode categorical variables

---

## 📚 4. Recommendation & Retrieval

### Topic: Matrix Factorization
**Implement:**
- `mf_sgd(R, k, lr, reg, epochs)`
- Predict missing ratings
- Compute RMSE on validation set

---

### Topic: TF-IDF + Cosine Similarity
**Implement:**
- Compute TF-IDF manually (without sklearn)
- Build document embeddings
- Implement `cosine_sim_topk(query_vec, doc_mat, k)`

---

### Topic: Retrieval Evaluation
**Implement:**
- `recall_at_k(y_true, y_pred, k)`
- `mrr_at_k(y_true, y_pred, k)`
- `map_at_k(y_true, y_pred, k)`

---

## 🧩 5. LLM / RAG Coding Topics (新近热门)

### Topic: Text Chunking & Embedding
**Implement:**
- `chunk_text(text, size=256, overlap=64)`
- Token-level sliding window chunking
- Optional deduplication

---

### Topic: Vector Retrieval
**Implement:**
- `retrieve(query_emb, doc_embs, topk)` using cosine or inner product
- Normalize embeddings
- Add reranking with dot-product score

---

### Topic: Scaled Dot-Product Attention
**Implement:**
- `scaled_dot_product_attention(Q, K, V, mask=None)`
- Apply softmax with numerical stability
- Verify output shape and gradients

---

## ⚙️ 6. ML System Design Mini Tasks

### Topic: Train-Validate-Test Pipeline
**Implement:**
- Simple pipeline with `load_data → preprocess → train → evaluate`
- Add early stopping & learning rate decay
- Support CLI arguments (use `argparse`)

---

### Topic: Online Prediction / AB Testing
**Implement:**
- Streaming prediction simulator
- Log results to JSON/CSV
- Compute online metrics (CTR lift, precision@k)

---

## 🧰 7. Python Engineering Basics

### Topic: Coding Style & Reliability
**Implement:**
- `@dataclass` for model configs
- Custom exceptions for training errors
- Unit tests with `pytest`
- Logging setup (INFO/DEBUG levels)
- Parallel data processing using `concurrent.futures`

---

# 📅 Recommended Practice Order (1 Week)

| Day | Focus Topic | Task |
|-----|--------------|------|
| 1 | Linear & Logistic Regression | Implement + test loss convergence |
| 2 | KMeans & PCA | Visualize convergence and reconstruction error |
| 3 | Neural Net (2-layer) | Gradient check + backprop |
| 4 | Metrics & CV | ROC-AUC, k-fold CV |
| 5 | Recommendation / Retrieval | Cosine sim + recall@k |
| 6 | RAG / Attention | Implement chunking + attention |
| 7 | Pipeline & Engineering | End-to-end script + unit tests |

---

# 🧠 Advanced Deep Learning Topics — From Scratch

以下内容延续之前的结构，聚焦于深度学习方向常被考察的“手写核心组件”题。

---

## 🧩 8. Convolutional Neural Networks (CNN)

### Topic: 2D Convolution (no frameworks)
**Implement:**
- `conv2d(X, W, stride=1, padding=0)`
  - X: input (N, C_in, H, W)
  - W: weights (C_out, C_in, KH, KW)
- Support zero padding and stride
- Verify output shape correctness
- Compare with `scipy.signal.correlate2d` for validation

---

### Topic: Convolutional Layer + Activation
**Implement:**
- `Conv2D` layer with ReLU activation
- Forward + Backward propagation (compute gradients wrt X and W)
- Gradient check with numerical diff
- Optional: Batch normalization layer (with running mean/var)

---

### Topic: CNN Mini Model
**Implement:**
- Simple CNN for MNIST-like data:
  - `Conv2D → ReLU → MaxPool → Flatten → Linear → Softmax`
- Use SGD training loop
- Evaluate accuracy and visualize filters

---

## 🧬 9. Graph Neural Networks (GNN)

### Topic: Graph Representation
**Implement:**
- Load adjacency matrix `A` and feature matrix `X`
- Normalize adjacency:  
  `A_hat = D^{-1/2} (A + I) D^{-1/2}`

---

### Topic: Graph Convolution Layer
**Implement:**
- `graph_conv(A_hat, X, W)`
  - `Z = A_hat @ X @ W`
  - Add ReLU activation
- Verify with small graph (e.g., 4 nodes, 2 features)

---

### Topic: 2-layer GCN Model
**Implement:**
- `GCN(A, X, W1, W2)`  
  - `H = ReLU(A_hat @ X @ W1)`
  - `Z = softmax(A_hat @ H @ W2)`
- Train on toy dataset for node classification (e.g., 3 classes)
- Optional: Add dropout or skip connections

---

### Topic: Message Passing Framework (Generalized GNN)
**Implement:**
- `message_passing(nodes, edges, update_fn, aggregate_fn)`
- Support:
  - Edge-level messages
  - Node updates by aggregation (mean/sum)
- Demonstrate on simple graph task (e.g., degree prediction)

---

## ⚡ 10. Transformer Building Blocks

### Topic: Scaled Dot-Product Attention
**Implement:**
- `attention(Q, K, V, mask=None)`
  - Compute attention weights: `softmax(QK^T / sqrt(d_k))`
  - Apply to V
- Ensure numerical stability (subtract max before softmax)

---

### Topic: Multi-Head Attention
**Implement:**
- `multi_head_attention(Q, K, V, num_heads, d_model)`
  - Split Q, K, V into heads
  - Apply attention per head
  - Concatenate and project
- Verify output shape `(batch, seq_len, d_model)`

---

### Topic: Positional Encoding
**Implement:**
- Sinusoidal encoding:
  - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
- Plot first few dimensions to visualize pattern

---

### Topic: Transformer Encoder Block
**Implement:**
- Components:
  - Multi-head attention
  - LayerNorm
  - FeedForward (2-layer MLP)
  - Residual connections
- Structure:
