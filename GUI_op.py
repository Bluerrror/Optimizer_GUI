import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# ---------------- Safe eval -----------------
def safe_eval(expr, x=None, y=None):
    allowed = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt, "pi": np.pi}
    return eval(expr, {"__builtins__": {}}, {"x": x, "y": y, **allowed})

# ---------------- Default functions -----------------
DEFAULT_1D = "np.sin(5*x) + np.sin(2*x) + 0.1*x**2"
DEFAULT_2D = "np.sin(5*x) + np.sin(5*y) + 0.1*(x**2 + y**2)"

# ---------------- Genetic Algorithm -----------------
def run_ga(func_input, is_2D, bounds, n_iter=50, pop_size=30, x0=None):
    if is_2D:
        xL, xU, yL, yU = bounds
        pop = np.random.uniform([xL,yL],[xU,yU],size=(pop_size,2))
    else:
        xL, xU = bounds
        pop = np.random.uniform(xL,xU,size=(pop_size,1))
    
    history, best_history = [], []
    best_val = float("inf")
    best_pos = None
    
    for _ in range(n_iter):
        if is_2D:
            vals = np.array([safe_eval(func_input,x=p[0],y=p[1]) for p in pop])
        else:
            vals = np.array([safe_eval(func_input,x=p[0]) for p in pop])
        
        idx = np.argmin(vals)
        if vals[idx] < best_val:
            best_val = vals[idx]
            best_pos = pop[idx].copy()
        
        if is_2D:
            history.append((best_pos[0],best_pos[1],best_val))
        else:
            history.append((best_pos[0],best_val))
        best_history.append(best_val)
        
        elite = pop[idx:idx+1]
        pop = elite + 0.1*np.random.randn(pop_size,*elite.shape[1:])
        if is_2D:
            pop[:,0] = np.clip(pop[:,0],xL,xU)
            pop[:,1] = np.clip(pop[:,1],yL,yU)
        else:
            pop[:,0] = np.clip(pop[:,0],xL,xU)
    
    return history, best_history, best_pos, best_val

# ---------------- SciPy Optimization -----------------
def run_scipy(func_input, is_2D, bounds, method="BFGS", x0=None, max_iter=50):
    history = []
    
    if is_2D:
        xL, xU, yL, yU = bounds
        if x0 is None:
            x0 = np.random.uniform([xL, yL], [xU, yU])
        
        def func(p):
            val = safe_eval(func_input, x=p[0], y=p[1])
            if len(history) < max_iter:
                history.append((p[0], p[1], val))
            return val
        
        res = minimize(func, x0, method=method, bounds=((xL,xU),(yL,yU)) if method in ["L-BFGS-B","TNC","SLSQP"] else None,
                       options={"maxiter": max_iter})
        best_pos = res.x
        best_val = res.fun
        nfev = res.nfev
    else:
        xL, xU = bounds
        if x0 is None:
            x0 = np.random.uniform(xL, xU)
        
        def func(x):
            val = safe_eval(func_input, x=x[0])
            if len(history) < max_iter:
                history.append((x[0], val))
            return val
        
        res = minimize(func, [x0], method=method, bounds=((xL,xU),) if method in ["L-BFGS-B","TNC","SLSQP"] else None,
                       options={"maxiter": max_iter})
        best_pos = res.x[0]
        best_val = res.fun
        nfev = res.nfev
    
    best_history = [h[-1] for h in history]
    return history, best_history, best_pos, best_val, nfev

# ---------------- Main App -----------------
st.set_page_config(layout="wide")
st.title("🔬 Interactive Optimization Playground")

# --- Sidebar Inputs ---
with st.sidebar:
    mode = st.radio("Choose mode:", ["1D", "2D"], horizontal=False)
    is_2D = (mode=="2D")
    
    func_input = st.text_input("Objective Function:", DEFAULT_2D if is_2D else DEFAULT_1D)
    
    if is_2D:
        x_lower, x_upper = st.slider("x range", -5.0, 5.0, (-3.0,3.0))
        y_lower, y_upper = st.slider("y range", -5.0, 5.0, (-3.0,3.0))
        bounds = (x_lower,x_upper,y_lower,y_upper)
    else:
        x_lower, x_upper = st.slider("x range", -10.0, 10.0, (-6.0,6.0))
        bounds = (x_lower,x_upper)
    
    optimizer = st.selectbox("Optimization Method:", ["Genetic Algorithm (GA)","BFGS", "Nelder-Mead", "Powell", "CG", "L-BFGS-B", "TNC", "SLSQP"])
    max_iter = st.number_input("Maximum iterations", min_value=1, max_value=1000, value=50, step=1)
    
    if is_2D:
        x_init = st.number_input("Initial x", min_value=float(x_lower), max_value=float(x_upper), value=0.0)
        y_init = st.number_input("Initial y", min_value=float(y_lower), max_value=float(y_upper), value=0.0)
        x0 = [x_init, y_init]
    else:
        x0 = st.number_input("Initial x", min_value=float(x_lower), max_value=float(x_upper), value=0.0)

# --- Run Optimization ---
if st.sidebar.button("🚀 Run Optimization"):
    if optimizer=="Genetic Algorithm (GA)":
        history, best_history, best_pos, best_val = run_ga(func_input,is_2D,bounds,n_iter=max_iter,x0=x0)
        nfev = len(history)
        method_used = "GA"
    else:
        history, best_history, best_pos, best_val, nfev = run_scipy(func_input,is_2D,bounds,optimizer,x0,max_iter)
        method_used = optimizer
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        if is_2D:
            x_grid = np.linspace(x_lower,x_upper,80)
            y_grid = np.linspace(y_lower,y_upper,80)
            X,Y = np.meshgrid(x_grid,y_grid)
            Z = np.vectorize(lambda x,y: safe_eval(func_input,x,y))(X,Y)
            
            fig = make_subplots(rows=1,cols=2,specs=[[{"type":"contour"},{"type":"surface"}]],
                                subplot_titles=("Contour","3D Surface"))
            # Background
            fig.add_trace(go.Contour(z=Z,x=x_grid,y=y_grid,colorscale="Viridis",
                                     contours=dict(showlabels=True),showscale=False),1,1)
            fig.add_trace(go.Surface(z=Z,x=X,y=Y,colorscale="Viridis",showscale=False,opacity=0.8),1,2)
            # Paths
            fig.add_trace(go.Scatter(x=[history[0][0]], y=[history[0][1]], mode="lines+markers",
                                     marker=dict(color="red",size=6), name="Path"),1,1)
            fig.add_trace(go.Scatter3d(x=[history[0][0]], y=[history[0][1]], z=[history[0][2]], mode="lines+markers",
                                       marker=dict(color="red",size=4), line=dict(color="red"), name="Path3D"),1,2)
            
            # Slider steps (update only path traces 2 & 3)
            steps=[]
            for i in range(len(history)):
                xs = [h[0] for h in history[:i+1]]
                ys = [h[1] for h in history[:i+1]]
                zs = [h[2] for h in history[:i+1]]
                step = dict(
                    method="restyle",
                    args=[{"x":[xs,xs], "y":[ys,ys], "z":[None,zs]}, [2,3]],
                    label=str(i+1)
                )
                steps.append(step)
            sliders=[dict(active=0, currentvalue={"prefix":"Iteration: "}, steps=steps)]
            fig.update_layout(sliders=sliders, title="Optimization Progress", height=600)
            st.plotly_chart(fig,use_container_width=True)
        
        else:
            x_grid = np.linspace(x_lower,x_upper,500)
            y_vals = [safe_eval(func_input,x) for x in x_grid]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_grid, y=y_vals, mode="lines", name="Function"))
            fig.add_trace(go.Scatter(x=[history[0][0]], y=[history[0][1]], mode="lines+markers",
                                     marker=dict(color="red",size=6), name="Path"))
            
            # Slider steps
            steps=[]
            for i in range(len(history)):
                xs = [h[0] for h in history[:i+1]]
                ys = [h[1] for h in history[:i+1]]
                step = dict(
                    method="restyle",
                    args=[{"x":[x_grid,xs], "y":[y_vals,ys]}],
                    label=str(i+1)
                )
                steps.append(step)
            sliders=[dict(active=0, currentvalue={"prefix":"Iteration: "}, steps=steps)]
            fig.update_layout(sliders=sliders, title="Optimization Progress", height=500)
            st.plotly_chart(fig,use_container_width=True)
        
        # Convergence curve
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=best_history, mode="lines+markers", name="Best f"))
        fig_conv.update_layout(title="Convergence Curve", xaxis_title="Iteration", yaxis_title="Best f")
        st.plotly_chart(fig_conv,use_container_width=True)
    
    with col2:
        st.success(f"✅ Best solution: {best_pos} with f = {best_val:.4f}")
        st.info(f"**Optimizer/Method:** {method_used} | **Function evaluations:** {nfev} | **Initial value:** {x0} | **Iterations:** {len(history)}")
        df = pd.DataFrame(history, columns=["x","f(x)"] if not is_2D else ["x","y","f(x,y)"])
        st.subheader("Iteration History")
        st.dataframe(df)
        
# --- Footer / About Me ---
st.markdown("---")
st.markdown(
    """
    **Author:** Shahab Shojaeezadeh  
    **Email:** shahab@uni-kassel.de  
    **University:** University of Kassel
    """
)
