# 🔬 Interactive Optimization Playground

**Streamlit App | Python | SciPy**  
**License:** MIT  
**Made by:** Shahab Shojaeezadeh  

A dynamic, interactive web application built with **Streamlit** and **SciPy** to visualize and compare optimization algorithms on custom 1D and 2D objective functions. Perfect for researchers, students, and enthusiasts exploring global and local optimization techniques!

Watch optimization unfold in real-time with animated paths, convergence curves, and iteration histories. From classic gradients to evolutionary strategies—optimize with ease!

---

## ✨ Features

- **Dual Modes:** Seamlessly switch between **1D** (line plots) and **2D** (contour + 3D surface) visualizations.  
- **Custom Functions:** Define your own objective functions using safe mathematical expressions  
  *(e.g., `np.sin(5*x) + np.sin(5*y) + 0.1*(x**2 + y**2)`).*  
- **Rich Optimizers:**
  - **Differential Evolution** (global, population-based via SciPy)
  - **Gradient-based:** BFGS, CG, L-BFGS-B
  - **Derivative-free:** Nelder-Mead, Powell, TNC, SLSQP  
- **Interactive Visuals:** Animated sliders to trace optimization paths iteration-by-iteration, real-time convergence plots, and detailed history tables.  
- **User Controls:** Adjustable bounds, initial conditions, and iteration limits.  
- **Safe & Secure:** Expression evaluation sandboxed to prevent code injection.  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+  
- Git  

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/interactive-optimization-playground.git
cd interactive-optimization-playground
