import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution
# ---------------- Safe eval -----------------
def safe_eval(expr, x=None, y=None):
    allowed = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt, "pi": np.pi}
    return eval(expr, {"__builtins__": {}}, {"x": x, "y": y, **allowed})
# ---------------- Default functions -----------------
DEFAULT_1D = "np.sin(5*x) + np.sin(2*x) + 0.1*x**2"
DEFAULT_2D = "np.sin(5*x) + np.sin(5*y) + 0.1*(x**2 + y**2)"
# ---------------- Differential Evolution -----------------
def run_de(func_input, is_2D, bounds, max_iter=50, x0=None):
    history = []
   
    popsize = 15
    if is_2D:
        xL, xU, yL, yU = bounds
        bounds_list = [(xL, xU), (yL, yU)]
        low = np.array([xL, yL])
        high = np.array([xU, yU])
        ndim = 2
       
        def func(p):
            return safe_eval(func_input, x=p[0], y=p[1])
       
        def callback(xk, convergence=1):
            val = func(xk)
            history.append((xk[0], xk[1], val))
    else:
        xL, xU = bounds
        bounds_list = [(xL, xU)]
        low = np.array([xL])
        high = np.array([xU])
        ndim = 1
       
        def func(p):
            return safe_eval(func_input, x=p[0])
       
        def callback(xk, convergence=1):
            val = func(xk)
            history.append((xk[0], val))
   
    # Generate initial population
    init_pop = np.random.uniform(low, high, (popsize, ndim))
    if x0 is not None:
        x0_arr = np.atleast_1d(x0)
        if len(x0_arr) == ndim:
            init_pop[0] = np.clip(x0_arr, low, high)
   
    # Evaluate initial population and add best to history
    init_vals = np.array([func(ind) for ind in init_pop])
    init_idx = np.argmin(init_vals)
    init_best_pos = init_pop[init_idx]
    init_best_val = init_vals[init_idx]
    if is_2D:
        history.append((init_best_pos[0], init_best_pos[1], init_best_val))
    else:
        history.append((init_best_pos[0], init_best_val))
   
    # Run DE
    res = differential_evolution(func, bounds_list, maxiter=max_iter, callback=callback,
                                 init=init_pop, popsize=popsize, tol=0.01, seed=42)
    best_pos = res.x
    best_val = res.fun
    nfev = res.nfev
   
    # Compute best_history as cumulative min
    vals = [h[-1] for h in history]
    best_history = [min(vals[:i+1]) for i in range(len(vals))]
    return history, best_history, best_pos, best_val, nfev
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
   
    optimizer = st.selectbox("Optimization Method:", ["Differential Evolution", "BFGS", "Nelder-Mead", "Powell", "CG", "L-BFGS-B", "TNC", "SLSQP"])
    max_iter = st.number_input("Maximum iterations", min_value=1, max_value=1000, value=50, step=1)
   
    if is_2D:
        x_init = st.number_input("Initial x", min_value=float(x_lower), max_value=float(x_upper), value=0.0)
        y_init = st.number_input("Initial y", min_value=float(y_lower), max_value=float(y_upper), value=0.0)
        x0 = [x_init, y_init]
    else:
        x0 = st.number_input("Initial x", min_value=float(x_lower), max_value=float(x_upper), value=0.0)
# --- Run Optimization ---
if st.sidebar.button("🚀 Run Optimization"):
    if optimizer=="Differential Evolution":
        history, best_history, best_pos, best_val, nfev = run_de(func_input,is_2D,bounds,max_iter=max_iter,x0=x0)
        method_used = "Differential Evolution"
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
            # Path traces empty initially
            path_trace2 = go.Scatter(x=[], y=[], mode="lines+markers",
                                     marker=dict(color="red",size=6), name="Path", showlegend=True)
            path_trace3 = go.Scatter3d(x=[], y=[], z=[], mode="lines+markers",
                                       marker=dict(color="red",size=4), line=dict(color="red"), name="Path3D", showlegend=False)
            fig.add_trace(path_trace2,1,1)
            fig.add_trace(path_trace3,1,2)
            frames = []
            if history:
                frame_list = []
                for i in range(len(history)):
                    xs = [h[0] for h in history[:i+1]]
                    ys = [h[1] for h in history[:i+1]]
                    zs = [h[2] for h in history[:i+1]]
                    frame_data2 = go.Scatter(x=xs, y=ys, mode="lines+markers",
                                             marker=dict(color="red",size=6))
                    frame_data3 = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers",
                                               marker=dict(color="red",size=4), line=dict(color="red"))
                    frame = go.Frame(data=[None, None, frame_data2, frame_data3], name=str(i+1))
                    frame_list.append(frame)
                if frame_list:
                    first = frame_list[0]
                    fig.data[2].x = first.data[2].x
                    fig.data[2].y = first.data[2].y
                    fig.data[3].x = first.data[3].x
                    fig.data[3].y = first.data[3].y
                    fig.data[3].z = first.data[3].z
                    frames = frame_list[1:]
           
            # Sliders
            steps = [dict(method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                          label="1")]
            for k in range(len(frames)):
                step = dict(method="animate",
                            args=[[k], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate",
                                        "transition": {"duration": 0}}],
                            label=str(k + 2))
                steps.append(step)
            sliders = [dict(active=0, currentvalue={"prefix": "Iteration: "}, steps=steps)]
           
            # Updatemenus
            updatemenus = []
            if len(steps) > 1:
                updatemenus = [dict(type="buttons",
                                    buttons=[dict(label="Play",
                                                  method="animate",
                                                  args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   transition=dict(duration=300),
                                                                   fromcurrent=True, mode="immediate")]),
                                             dict(label="Pause",
                                                  method="animate",
                                                  args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                     mode="immediate")])],
                                    direction="left",
                                    pad=dict(r=10, b=10),
                                    showactive=False,
                                    x=0.01,
                                    xanchor="left",
                                    y=0.01,
                                    yanchor="bottom")]
           
            fig.update_layout(sliders=sliders, updatemenus=updatemenus,
                              title="Optimization Progress", height=600)
            st.plotly_chart(fig,use_container_width=True)
       
        else:
            x_grid = np.linspace(x_lower,x_upper,500)
            y_vals = [safe_eval(func_input,x) for x in x_grid]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_grid, y=y_vals, mode="lines", name="Function"))
            # Path trace empty
            path_trace = go.Scatter(x=[], y=[], mode="lines+markers",
                                    marker=dict(color="red",size=6), name="Path")
            fig.add_trace(path_trace)
            frames = []
            if history:
                frame_list = []
                for i in range(len(history)):
                    xs = [h[0] for h in history[:i+1]]
                    ys = [h[1] for h in history[:i+1]]
                    frame_data = go.Scatter(x=xs, y=ys, mode="lines+markers",
                                            marker=dict(color="red",size=6))
                    frame = go.Frame(data=[None, frame_data], name=str(i+1))
                    frame_list.append(frame)
                if frame_list:
                    first = frame_list[0]
                    fig.data[1].x = first.data[1].x
                    fig.data[1].y = first.data[1].y
                    frames = frame_list[1:]
           
            # Sliders
            steps = [dict(method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                          label="1")]
            for k in range(len(frames)):
                step = dict(method="animate",
                            args=[[k], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate",
                                        "transition": {"duration": 0}}],
                            label=str(k + 2))
                steps.append(step)
            sliders = [dict(active=0, currentvalue={"prefix": "Iteration: "}, steps=steps)]
           
            # Updatemenus
            updatemenus = []
            if len(steps) > 1:
                updatemenus = [dict(type="buttons",
                                    buttons=[dict(label="Play",
                                                  method="animate",
                                                  args=[None, dict(frame=dict(duration=500, redraw=True),
                                                                   transition=dict(duration=300),
                                                                   fromcurrent=True, mode="immediate")]),
                                             dict(label="Pause",
                                                  method="animate",
                                                  args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                     mode="immediate")])],
                                    direction="left",
                                    pad=dict(r=10, b=10),
                                    showactive=False,
                                    x=0.1,
                                    xanchor="left",
                                    y=0,
                                    yanchor="bottom")]
           
            fig.update_layout(sliders=sliders, updatemenus=updatemenus,
                              title="Optimization Progress", height=500)
            st.plotly_chart(fig,use_container_width=True)
       
        # Convergence curve
        if best_history:
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
