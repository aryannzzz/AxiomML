# AxiomML

**Building Machine Learning & Deep Learning from First Principles.**

AxiomML is an educational, from-scratch implementation of fundamental machine learning algorithms. The goal is to demystify the "black box" nature of libraries like Scikit-learn and PyTorch by providing clean, well-documented, and simple code for the core components of ML.

> **An axiom is a statement that is taken to be true, to serve as a premise or starting point for further reasoning and arguments.**
> This library is our starting point.

---

## 🧠 Philosophy

Modern ML libraries are powerful but abstract. They hide the beautiful math and logic underneath layers of optimization and API design. **AxiomML strips that away.** Each algorithm is built from the ground up using only NumPy and core Python, prioritizing clarity and educational value over performance and features.

**Core Principles:**
- **Transparency**: Every line of code is documented with the "why" behind it
- **Mathematical Rigor**: From theory to implementation with clear explanations
- **Simplicity**: Clean, readable code that mirrors the underlying mathematics
- **Educational First**: Built for learning, not production deployment

---

## 🏗️ Project Structure
```
AxiomML/
├── axiom/
│   ├── linear_model/       # Linear models (regression & classification)
│   ├── tree/               # Decision trees
│   ├── ensemble/           # Random Forests and ensemble methods
│   ├── svm/                # Support Vector Machines
│   ├── neighbors/          # K-Nearest Neighbors
│   ├── naive_bayes/        # Naive Bayes classifiers
│   ├── neural_networks/    # Deep learning components ✨ NEW
│   ├── preprocessing/      # Data preprocessing utilities
│   └── metrics/            # Evaluation metrics ✨ NEW
├── examples/               # Comprehensive usage examples
└── requirements.txt        # Project dependencies
```

---

## 📚 Implemented Algorithms

### ✅ Phase 1: Core Regression (COMPLETED)

| Algorithm | Description | Key Concepts |
|-----------|-------------|--------------|
| **Simple Linear Regression** | OLS and Gradient Descent | Least squares, optimization |
| **Multiple Linear Regression** | Matrix-based normal equation | Linear algebra, vectorization |
| **Polynomial Regression** | Feature transformation | Basis functions, overfitting |
| **Ridge Regression** | L2 regularization | Bias-variance tradeoff, shrinkage |
| **Support Vector Regression** | Epsilon-insensitive loss | Margin optimization, kernel trick |
| **Decision Tree Regressor** | Recursive splitting | MSE minimization, pruning |
| **Random Forest Regressor** | Ensemble of trees | Bagging, feature randomness |

### ✅ Phase 2: Core Classification (COMPLETED)

| Algorithm | Description | Key Concepts |
|-----------|-------------|--------------|
| **Logistic Regression** | Binary classification | Sigmoid, cross-entropy loss |
| **K-Nearest Neighbors** | Instance-based learning | Distance metrics, voting |
| **Support Vector Classifier** | Maximum margin | Hinge loss, kernel methods |
| **Naive Bayes** | Probabilistic classification | Bayes theorem, independence |
| **Decision Tree Classifier** | Information-based splitting | Gini impurity, entropy |
| **Random Forest Classifier** | Voting ensemble | Bootstrap aggregating |

### ✅ Phase 3: Deep Learning Foundations (COMPLETED - 60%)

#### Neural Network Core Components ✅ **COMPLETED**
- ✅ **Activation Functions** (Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, ELU)
- ✅ **Loss Functions** (MSE, MAE, BCE, CCE, Hinge, Huber)
- ✅ **Optimizers** (SGD, Momentum, RMSprop, Adam, AdaGrad)
- ✅ **Layer Types** (Dense, Dropout, BatchNorm, LayerNorm)
- ✅ **Initialization** (Xavier, He, LeCun, + variants)

#### Architectures (NEXT)
- [ ] **Feedforward Neural Networks** - Multi-layer perceptrons with backpropagation
- [ ] **Convolutional Neural Networks** - Conv2D, pooling, modern architectures
- [ ] **Recurrent Neural Networks** - Vanilla RNN, LSTM, GRU for sequences
- [ ] **Transformers** - Self-attention, multi-head attention, positional encoding

#### NLP Components (PLANNED)
- [ ] **Word Embeddings** - Word2Vec (CBOW, Skip-gram), GloVe
- [ ] **Text Preprocessing** - Tokenization, vocabulary, padding
- [ ] **Sequence Models** - Language modeling, classification, generation

### ✅ Phase 3.5: Evaluation Metrics (COMPLETED)

#### Classification Metrics ✅
- ✅ `accuracy_score` - Overall correctness
- ✅ `precision_score` - Positive prediction accuracy
- ✅ `recall_score` - Actual positive detection rate
- ✅ `f1_score` - Harmonic mean of precision & recall
- ✅ `confusion_matrix` - Prediction breakdown
- ✅ `classification_report` - Comprehensive metrics
- ✅ `balanced_accuracy_score` - For imbalanced data
- ✅ `matthews_corrcoef` - Correlation metric
- ✅ `log_loss` - For probability predictions

#### Regression Metrics ✅
- ✅ `mean_squared_error` (MSE) - Squared errors
- ✅ `root_mean_squared_error` (RMSE) - Square root of MSE
- ✅ `mean_absolute_error` (MAE) - Absolute errors
- ✅ `r2_score` - Variance explained (R²)
- ✅ `adjusted_r2_score` - R² adjusted for features
- ✅ `mean_absolute_percentage_error` (MAPE) - Percentage errors
- ✅ `median_absolute_error` - Robust to outliers
- ✅ `max_error` - Worst-case error
- ✅ `explained_variance_score` - Variance explained
- ✅ `mean_squared_log_error` (MSLE) - MSE in log space
- ✅ `regression_report` - Comprehensive metrics

### 🔮 Phase 4: Advanced ML (PLANNED)

<details>
<summary><b>Unsupervised Learning</b></summary>

- [ ] K-Means Clustering
- [ ] Hierarchical Clustering (Agglomerative, Divisive)
- [ ] DBSCAN
- [ ] Gaussian Mixture Models (GMM)
- [ ] Spectral Clustering

</details>

<details>
<summary><b>Dimensionality Reduction</b></summary>

- [ ] Principal Component Analysis (PCA)
- [ ] Linear Discriminant Analysis (LDA)
- [ ] t-SNE
- [ ] UMAP
- [ ] Autoencoders for reduction

</details>

<details>
<summary><b>Ensemble Methods</b></summary>

- [ ] Gradient Boosting Machines (GBM)
- [ ] XGBoost from scratch
- [ ] AdaBoost
- [ ] CatBoost concepts
- [ ] Stacking ensembles

</details>

<details>
<summary><b>Advanced Architectures</b></summary>

- [ ] Autoencoders (Vanilla, Variational, Convolutional)
- [ ] Generative Adversarial Networks (GANs)
- [ ] Graph Neural Networks (GNNs)
- [ ] Diffusion Models
- [ ] Reinforcement Learning basics (Q-Learning, DQN)

</details>

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/aryannzzz/AxiomML.git
cd AxiomML
pip install -r requirements.txt
```

### Basic Usage

The API mirrors Scikit-learn for familiarity:

#### Regression
```python
from axiom.linear_model import LinearRegression, PolynomialRegression
from axiom.ensemble import RandomForestRegressor
from axiom.metrics import mean_squared_error, r2_score

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
```

#### Classification
```python
from axiom.linear_model import LogisticRegression
from axiom.ensemble import RandomForestClassifier
from axiom.metrics import accuracy_score, classification_report

# Train model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
```

#### Neural Networks
```python
from axiom.neural_networks import Dense, ReLU, Sigmoid, Adam, BinaryCrossEntropy

# Build network
hidden = Dense(10, 64)
hidden.initialize('he')
output = Dense(64, 1)
output.initialize('xavier')

# Setup training
relu = ReLU()
sigmoid = Sigmoid()
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossEntropy()

# Training loop
for epoch in range(epochs):
    # Forward pass
    h1 = hidden(X, training=True)
    a1 = relu(h1)
    h2 = output(a1, training=True)
    y_pred = sigmoid(h2)
    
    # Compute loss
    loss = loss_fn(y_true, y_pred)
    
    # Backward pass and update
    # ... (see examples for complete implementation)
```

---

## 📖 Examples

Comprehensive examples for every module:

| Example File | Description | Examples |
|--------------|-------------|----------|
| `regression_examples.py` | Linear regression models | 5 examples |
| `classification_examples.py` | Classification algorithms | 7 examples |
| `ensemble_examples.py` | Random forests | 7 examples |
| `neural_networks_demo.py` | Neural network components | 7 examples |
| `metrics_examples.py` | Evaluation metrics | 8 examples |

**Total: 34 comprehensive examples!**

Run examples:
```bash
python examples/regression_examples.py
python examples/classification_examples.py
python examples/ensemble_examples.py
python examples/neural_networks_demo.py
python examples/metrics_examples.py
```

---

## 🎯 Educational Features

Each implementation includes:

- 📐 **Mathematical foundations** - Derivations and proofs in comments
- 🔍 **Algorithm walkthroughs** - Step-by-step execution flow
- 💡 **Design intuition** - Why certain choices were made
- ⚠️ **Edge cases** - Numerical stability and corner cases
- 📊 **Comparisons** - How it differs from production libraries
- 📚 **References** - Links to papers and textbooks

### Example: Understanding Gradient Descent
```python
# From axiom/linear_model/linear_regression.py

def _gradient_descent(self, X, y, learning_rate=0.01, n_iterations=1000):
    """
    Optimize weights using gradient descent.
    
    Mathematical Intuition:
    - We want to minimize J(w) = (1/2m) * Σ(h(x_i) - y_i)²
    - Gradient: ∇J(w) = (1/m) * X^T * (Xw - y)
    - Update rule: w := w - α * ∇J(w)
    
    Why this works:
    - The gradient points in the direction of steepest ascent
    - Moving opposite to the gradient minimizes the loss
    - Learning rate α controls step size
    """
    m = len(y)
    
    for i in range(n_iterations):
        # Forward pass: compute predictions
        predictions = X @ self.weights
        
        # Compute gradient of loss w.r.t weights
        gradient = (1/m) * X.T @ (predictions - y)
        
        # Update weights in the direction that decreases loss
        self.weights -= learning_rate * gradient
```

---

## 📊 Progress Tracker

| Phase | Component | Progress | Status |
|-------|-----------|----------|--------|
| 1️⃣ | Regression Algorithms | 7/7 | ✅ Complete |
| 2️⃣ | Classification Algorithms | 6/6 | ✅ Complete |
| 3️⃣ | Deep Learning Core | 5/5 | ✅ Complete |
| 3️⃣ | Neural Network Architectures | 0/4 | 🔄 Next |
| 3.5 | Evaluation Metrics | 20/20 | ✅ Complete |
| 4️⃣ | Advanced ML | 0/20+ | 🔮 Planned |

**Overall Completion: ~50%**

**Lines of Code:** ~8,000+ heavily documented lines

---

## 🤝 Contributing

This is primarily a personal educational journey, but contributions are welcome! Whether it's fixing bugs, improving documentation, or suggesting new algorithms.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Make your changes following our guidelines
4. Add tests and examples
5. Submit a pull request

### Development Guidelines

- **Clarity over performance** - Readable code beats optimized code
- **Document the "why"** - Explain mathematical intuition, not just implementation
- **Mirror sklearn API** - Maintain familiar `.fit()`, `.predict()`, `.score()` methods
- **Test thoroughly** - Add unit tests for edge cases
- **Provide examples** - Include usage demonstrations

---

## 📖 Learning Resources

Want to understand the math behind the algorithms? Check out these resources:

- [Mathematics for Machine Learning](https://mml-book.github.io/) - Linear algebra, calculus, probability
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Bishop's classic textbook
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow et al.
- [AxiomML Examples](./examples/) - Our own detailed walkthroughs

---

## 🙏 Acknowledgments

Inspired by the educational philosophy of:
- **3Blue1Brown** - Visualizing mathematics
- **Andrej Karpathy** - Making neural networks less scary
- **Fast.ai** - Practical deep learning for coders

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 💭 Philosophy

> "If you can't implement it from scratch, you don't truly understand it."

This isn't about reinventing the wheel. It's about understanding why the wheel is round.

---

<div align="center">

**[⭐ Star this repo](https://github.com/aryannzzz/AxiomML)** if you find it helpful!

Made with ❤️ for learners who want to peek under the hood

</div>
