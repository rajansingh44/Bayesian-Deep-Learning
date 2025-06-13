# Bayesian Inference in a One-Layer Neural Network

### Student Information
- **Name:** Rajan Vijaykumar Singh  
- **Matriculation Number:** 1567294  
- **Course:** Computational Intelligence  
- **Topic:** Bayesian Deep Learning and Uncertainty Estimation  

---

## 📌 Objective

This project demonstrates **Bayesian reasoning and uncertainty estimation** in a simple one-layer neural network using prior weight sampling. The predictive output is evaluated using two different activation functions: **Sigmoid** and **ReLU**.

The goal is to show how **uncertainty in weights** propagates to the **output prediction**, and how different activations affect the **distribution and behavior of predictions**.

---

## 🧠 Core Concepts

### Bayesian Deep Learning
- Traditional neural networks use fixed weights.
- In Bayesian deep learning, weights are treated as **random variables** with prior and posterior distributions.
- This approach allows the model to express **uncertainty** in predictions, which is crucial in safety-critical applications.

### Predictive Uncertainty
- Predictive distributions reflect the range of possible outputs given uncertain weights.
- Visualizing these distributions helps us understand how the model perceives **confidence** or **uncertainty**.

---

## 🧪 What the [Code](https://github.com/rajansingh44/Bayesian-Deep-Learning/blob/main/ReLU_and_Sigmoid_for_one_layer_network.py) Does

### Step-by-Step Overview

1. **Define a Prior**  
   - A Gaussian distribution N(0, 1) is used as the prior over weights.
.

2. **Sample Weights**  
   - 1000 samples are drawn to represent different possible weight values.

3. **Predict Outputs**  
   - For a fixed input \( x = 2.0 \), predictions are made using:
     - **Sigmoid activation**
     - **ReLU activation**

4. **Visualize Predictive Distributions**  
   - Histograms show how predicted outputs are distributed under each activation function.

---

## 📊 Output Interpretation : [Plots](https://github.com/rajansingh44/Bayesian-Deep-Learning/blob/main/Output_Plot.png)

### Sigmoid Activation
- Output is bounded between 0 and 1.
- Most predictions cluster around **0.5**, due to the sigmoid squashing effect.
- Smooth and symmetric predictive distribution.

### ReLU Activation
- Output is non-negative.
- Half the predictions are **zero** (due to ReLU cutting off negatives).
- Remaining predictions form a **long positive tail**, showing skewed uncertainty.

---

