# AxiomML

**Building Machine Learning & Deep Learning from First Principles.**

AxiomML is an educational, from-scratch implementation of fundamental machine learning algorithms. The goal is to demystify the "black box" nature of libraries like Scikit-learn and PyTorch by providing clean, well-documented, and simple code for the core components of ML.

> **An axiom is a statement that is taken to be true, to serve as a premise or starting point for further reasoning and arguments.**
> This library is our starting point.

## 🧠 Philosophy

Modern ML libraries are powerful but abstract. They hide the beautiful math and logic underneath layers of optimization and API design. **AxiomML strips that away.** Each algorithm is built from the ground up using only NumPy and core Python, prioritizing clarity and educational value over performance and features.

## 🏗️ Project Structure

AxiomML/
├── axiom/
│ ├── linear_model/ # Linear models (regression & classification)
│ ├── tree/ # Decision trees and ensembles
│ ├── ensemble/ # Random Forests and ensemble methods
│ ├── svm/ # Support Vector Machines
│ ├── neighbors/ # K-Nearest Neighbors
│ ├── naive_bayes/ # Naive Bayes classifiers
│ ├── preprocessing/ # Data preprocessing utilities
│ └── metrics/ # Evaluation metrics
├── examples/ # Jupyter notebooks with usage examples
├── tests/ # Unit tests for all implementations
└── requirements.txt # Project dependencies

## 📚 Implemented Algorithms (The Axioms)

### ✅ Phase 1: Core Regression Algorithms (COMPLETED)
- **Simple Linear Regression** - OLS and Gradient Descent implementations
- **Multiple Linear Regression** - Matrix-based normal equation solver
- **Polynomial Regression** - Feature transformation for nonlinear relationships
- **Ridge Regression** - L2 regularization for overfitting prevention
- **Support Vector Regression (SVR)** - Epsilon-insensitive loss with margin optimization
- **Decision Tree Regressor** - Recursive splitting with MSE minimization
- **Random Forest Regressor** - Ensemble of decorrelated decision trees

### ✅ Phase 2: Core Classification Algorithms (COMPLETED)
- **Logistic Regression** - Binary classification with cross-entropy loss
- **K-Nearest Neighbors (KNN)** - Instance-based learning with distance weighting
- **Support Vector Classifier (SVC)** - Maximum margin classification with hinge loss
- **Naive Bayes** - Probabilistic classification with Gaussian assumptions
- **Decision Tree Classifier** - Information gain/Gini impurity splitting
- **Random Forest Classifier** - Majority voting ensemble of trees

### 🔄 Phase 3: Deep Learning Foundations (IN PROGRESS)
#### Neural Network Core Components
- [ ] **Activation Functions** (Sigmoid, Tanh, ReLU, LeakyReLU, Softmax)
- [ ] **Loss Functions** (MSE, Cross-Entropy, Hinge, Huber)
- [ ] **Optimizers** (SGD, Momentum, Adam, RMSprop)
- [ ] **Layers** (Dense, Dropout, BatchNorm)

#### Neural Network Architectures
- [ ] **Feedforward Neural Networks (FNN)** - Backpropagation from scratch
- [ ] **Convolutional Neural Networks (CNN)** - Conv layers, pooling, modern architectures
- [ ] **Recurrent Neural Networks (RNN)** - LSTM, GRU, sequence modeling
- [ ] **Transformers** - Self-attention, multi-head attention, positional encoding

#### Natural Language Processing
- [ ] **Word Embeddings** (Word2Vec, GloVe from scratch)
- [ ] **Text Preprocessing** (Tokenization, padding, vocabulary building)
- [ ] **Sequence Models** - For text classification, generation, translation

### 🔮 Phase 4: Advanced ML & Specialized Architectures (PLANNED)
- **Unsupervised Learning**
  - [ ] K-Means Clustering
  - [ ] Hierarchical Clustering
  - [ ] DBSCAN
  - [ ] Gaussian Mixture Models

- **Dimensionality Reduction**
  - [ ] Principal Component Analysis (PCA)
  - [ ] Linear Discriminant Analysis (LDA)
  - [ ] t-SNE, UMAP

- **Ensemble Methods**
  - [ ] Gradient Boosting Machines (GBM)
  - [ ] XGBoost from scratch
  - [ ] AdaBoost

- **Advanced Architectures**
  - [ ] Autoencoders (Vanilla, Variational, Convolutional)
  - [ ] Generative Adversarial Networks (GANs)
  - [ ] Graph Neural Networks (GNNs)

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/a-jacked-nerd/AxiomML.git
    cd AxiomML
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Import and use!** The goal is to mirror the Scikit-learn API for familiarity.
    ```python
    # Regression Example
    from axiom.linear_model import LinearRegression
    from axiom.tree import DecisionTreeRegressor
    from axiom.ensemble import RandomForestRegressor

    # Classification Example  
    from axiom.linear_model import LogisticRegression
    from axiom.svm import SVC
    from axiom.ensemble import RandomForestClassifier

    # Model usage follows sklearn pattern
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    ```

## 🎯 Educational Value

Each implementation includes:
- **Detailed mathematical explanations** in comments
- **Step-by-step algorithm walkthroughs**
- **Intuition behind design choices**
- **Edge cases and numerical stability considerations**
- **Comparison with standard library implementations**

## 🔬 Example: Understanding Backpropagation

```python
# Coming soon in neural_networks/
from axiom.neural_networks.layers import Dense
from axiom.neural_networks.activations import ReLU, Sigmoid
from axiom.neural_networks.losses import CrossEntropy
from axiom.neural_networks.optimizers import Adam

# Build a neural network from fundamental components
model = Sequential([
    Dense(128, input_dim=784, activation=ReLU()),
    Dense(64, activation=ReLU()),
    Dense(10, activation=Sigmoid())
])

# Train with understanding of every operation
model.compile(loss=CrossEntropy(), optimizer=Adam())
model.fit(X_train, y_train, epochs=10, verbose=True)
```

🤝 Contributing
This is primarily a personal educational journey, but discussions, suggestions, and contributions are welcome! Feel free to open an issue to discuss a new algorithm or submit a pull request.

Development Guidelines:
Prioritize clarity over performance

Include comprehensive comments explaining the "why"

Maintain sklearn-like API where possible

Add examples and unit tests for new implementations

📊 Progress Tracking
✅ Regression Algorithms: 100% Complete

✅ Classification Algorithms: 100% Complete

🔄 Deep Learning Core: 0% Complete

🔮 Advanced ML: 0% Complete

📝 License
This project is open source and available under the MIT License.

"If you can't implement it from scratch, you don't truly understand it." - AxiomML Philosophy


## Key Updates Made:

1. **Current Progress**: Moved Regression and Classification algorithms to "COMPLETED" status
2. **Detailed Structure**: Added comprehensive project structure showing all modules
3. **Deep Learning Roadmap**: 
   - Core components (activations, losses, optimizers, layers)
   - Neural network architectures (FNN, CNN, RNN, Transformers)
   - NLP pipeline from scratch
4. **Advanced ML**: Added future categories for unsupervised learning, dimensionality reduction, and specialized architectures
5. **Educational Focus**: Emphasized the learning value with specific examples
6. **Progress Tracking**: Visual progress indicators for different phases
7. **Development Guidelines**: Clear contribution standards maintaining the educational focus

This README now accurately reflects our substantial progress while clearly outlining the exciting deep learning journey ahead! The structure sets us up perfectly for building neural networks from the ground up, starting with fundamental components and building toward complex architectures.
