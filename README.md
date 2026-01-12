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
Create a virtual environment (recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
requirements.txt:

shell
Copy code
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
scipy>=1.11.0
Running the App
Launch the app with:

bash
Copy code
streamlit run app.py
Open your browser at http://localhost:8501. Experiment away! 🎯

📖 Usage Guide
Sidebar Controls
Mode: Select "1D" or "2D" for dimensionality.

Objective Function: Enter a math expression (uses NumPy for trig, exp, etc.).

Bounds: Slider for variable ranges (e.g., x: -10 to 10).

Optimizer: Choose from the dropdown (e.g., "Differential Evolution").

Iterations: Set max iterations (default: 50).

Initial Point: Provide starting values (used where applicable).

Main Interface
Run Optimization button triggers the magic.

Left Column:

1D: Function line + path animation

2D: Contour plot + 3D surface with overlaid optimization trajectory

Convergence curve below

Right Column:

Best solution summary, stats, and iteration DataFrame

Example Functions
1D Default: np.sin(5*x) + np.sin(2*x) + 0.1*x**2 (multi-modal with quadratic trend)

2D Default: np.sin(5*x) + np.sin(5*y) + 0.1*(x**2 + y**2) (sinusoidal bowl)

Pro Tip: For global minima, try Differential Evolution on rugged landscapes!

📸 Screenshots
2D Mode: Contour & Surface with Path

2D Optimization

1D Mode: Line Plot with Convergence

Iteration History Table

(Add your own GIFs/screenshots to screenshots/ folder for GitHub rendering!)

🔧 Optimization Methods
Method	Type	Supports Bounds	Derivative-Free	Best For
Differential Evolution	Global (Evolutionary)	✅	✅	Multi-modal, noisy functions
BFGS	Local (Quasi-Newton)	❌	❌	Smooth, differentiable
Nelder-Mead	Local (Simplex)	❌	✅	Non-differentiable
Powell	Local (Conjugate)	❌	✅	Low-dimensional
CG	Local (Conjugate Gradient)	❌	❌	Large-scale, quadratic
L-BFGS-B	Local (Limited-Memory)	✅	❌	Constrained, memory-efficient
TNC	Local (Truncated Newton)	✅	❌	Constrained
SLSQP	Local (Sequential QP)	✅	❌	Equality/inequality constraints

Note: Initial conditions are incorporated into DE's population for a "warm start."

🛠️ Contributing
Contributions are welcome! 💖

Fork the repo

Create a feature branch:

bash
Copy code
git checkout -b feature/amazing-feature
Commit changes:

bash
Copy code
git commit -m 'Add amazing feature'
Push to branch:

bash
Copy code
git push origin feature/amazing-feature
Open a Pull Request

Please adhere to PEP 8 for code style. Issues and feature requests are also appreciated—let's optimize together!

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

👨‍💻 Author
Shahab Shojaeezadeh
PhD Candidate in Optimization & Machine Learning
📧 shahab@uni-kassel.de
🏛️ University of Kassel, Germany
