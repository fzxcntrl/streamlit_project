import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json

st.set_page_config(layout="wide", page_title="NeuralLab | Pro Optimizer")

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.w1 = 0.0
    st.session_state.w2 = 0.0
    st.session_state.b = 0.0
    st.session_state.loss_history = []
    st.session_state.log_history = []
    st.session_state.epoch_count = 0
    st.session_state.is_training = False
    st.session_state.current_mse = 0.0
    st.session_state.theme = "Dark"

with st.sidebar:
    st.markdown('<div style="font-weight: 900; font-size: 2em; text-transform: uppercase; letter-spacing: 1px;">Config</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("### Interface Theme")
    theme_selection = st.radio("UI Mode", ["Dark", "Light"], horizontal=True, label_visibility="collapsed")
    if theme_selection != st.session_state.theme:
        st.session_state.theme = theme_selection
        st.rerun()

if st.session_state.theme == "Dark":
    bg_color = "#0e1117"
    text_color = "#ffffff"
    primary = "#00ffff"
    secondary = "#ff00ff"
    glass_bg = "rgba(20, 20, 30, 0.6)"
    border_color = "rgba(255, 255, 255, 0.1)"
    term_bg = "#0a0a0f"
    term_text = "#00ffcc"
    grid_color = "rgba(255, 255, 255, 0.1)"
    btn_text = "#000000"
    tab_bg = "rgba(255,255,255,0.05)"
    tab_selected_bg = "rgba(0, 255, 255, 0.1)"
    shadow = "rgba(0, 0, 0, 0.5)"
else:
    bg_color = "#f4f6f9"
    text_color = "#111827"
    primary = "#0066cc"
    secondary = "#ff3366"
    glass_bg = "rgba(255, 255, 255, 0.85)"
    border_color = "rgba(0, 0, 0, 0.1)"
    term_bg = "#e8ecef"
    term_text = "#0369a1"
    grid_color = "rgba(0, 0, 0, 0.1)"
    btn_text = "#ffffff"
    tab_bg = "rgba(0,0,0,0.05)"
    tab_selected_bg = "rgba(0, 102, 204, 0.1)"
    shadow = "rgba(0, 0, 0, 0.05)"

st.markdown(f"""
<style>
    :root {{
        --primary: {primary};
        --secondary: {secondary};
        --bg-glass: {glass_bg};
        --border-glass: {border_color};
        --text-color: {text_color};
        --bg-main: {bg_color};
    }}
    .stApp {{
        background-color: var(--bg-main);
        transition: background-color 0.5s ease;
    }}
    p, h1, h2, h3, h4, h5, h6, span, label, div {{
        color: var(--text-color) !important;
        transition: color 0.5s ease;
    }}
    div.stButton > button {{
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 6px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: {btn_text} !important;
        font-weight: 900;
        border: none;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.75rem;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0px 8px 15px rgba(128, 128, 128, 0.3);
    }}
    div.stDownloadButton > button {{
        background: transparent;
        border: 1px solid var(--primary);
        color: var(--primary) !important;
    }}
    div.stDownloadButton > button:hover {{
        background: {tab_selected_bg};
    }}
    .main-title {{
        background: -webkit-linear-gradient(45deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5em;
        letter-spacing: -1px;
        margin-bottom: 0;
        padding-bottom: 0;
        display: inline-block;
    }}
    .sub-title {{
        font-family: monospace;
        font-size: 1.1em;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
        opacity: 0.7;
    }}
    .glass-panel {{
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid var(--border-glass);
        box-shadow: 0 8px 32px {shadow};
        margin-bottom: 24px;
        transition: all 0.5s ease;
    }}
    .terminal-window {{
        background-color: {term_bg};
        border: 1px solid var(--border-glass);
        border-radius: 8px;
        padding: 15px;
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        font-size: 0.85em;
        line-height: 1.6;
        height: 300px;
        overflow-y: hidden;
        transition: all 0.5s ease;
    }}
    .terminal-window span {{
        color: {term_text} !important;
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
    }}
    [data-testid="stMetricValue"] {{
        font-family: monospace;
        font-size: 2.2rem;
        color: var(--text-color) !important;
        font-weight: bold;
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--primary) !important;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {tab_bg};
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_selected_bg} !important;
        border-top: 2px solid var(--primary) !important;
        border-left: 1px solid var(--border-glass) !important;
        border-right: 1px solid var(--border-glass) !important;
    }}
    hr {{
        border-color: var(--border-glass);
    }}
</style>
""", unsafe_allow_html=True)

def generate_dataset(data_type, num_samples, noise_level):
    np.random.seed(42)
    if data_type == "Linear":
        x1 = np.random.uniform(-5, 5, num_samples)
        x2 = np.random.uniform(-5, 5, num_samples)
        y = 2.0 * x1 - 1.5 * x2 + 1.0 + np.random.normal(0, noise_level, num_samples)
        return x1, x2, y
    elif data_type == "Blobs (Classification)":
        x1_class0 = np.random.normal(-2, noise_level, num_samples // 2)
        x2_class0 = np.random.normal(-2, noise_level, num_samples // 2)
        x1_class1 = np.random.normal(2, noise_level, num_samples // 2)
        x2_class1 = np.random.normal(2, noise_level, num_samples // 2)
        x1 = np.concatenate([x1_class0, x1_class1])
        x2 = np.concatenate([x2_class0, x2_class1])
        y = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])
        return x1, x2, y
    elif data_type == "Moons (Classification)":
        t = np.linspace(0, np.pi, num_samples // 2)
        x1_0 = np.cos(t) + np.random.normal(0, noise_level/5, num_samples//2)
        x2_0 = np.sin(t) + np.random.normal(0, noise_level/5, num_samples//2)
        t2 = np.linspace(0, np.pi, num_samples // 2)
        x1_1 = 1 - np.cos(t2) + np.random.normal(0, noise_level/5, num_samples//2)
        x2_1 = 1 - np.sin(t2) - 0.5 + np.random.normal(0, noise_level/5, num_samples//2)
        x1 = np.concatenate([x1_0, x1_1])
        x2 = np.concatenate([x2_0, x2_1])
        y = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])
        return x1, x2, y

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def calculate_predictions(x1, x2, w1, w2, b, task_type):
    linear_comb = w1 * x1 + w2 * x2 + b
    if task_type == "Classification":
        return sigmoid(linear_comb)
    return linear_comb

def calculate_loss(y_true, y_pred, task_type, w1, w2, l2_lambda):
    l2_penalty = (l2_lambda / (2 * len(y_true))) * (w1**2 + w2**2)
    if task_type == "Regression":
        base_loss = np.mean((y_true - y_pred) ** 2)
    else:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        base_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return base_loss + l2_penalty

def compute_gradients(x1, x2, y_true, y_pred, w1, w2, l2_lambda):
    m = len(y_true)
    dz = y_pred - y_true
    dw1 = (1/m) * np.dot(dz, x1) + (l2_lambda/m) * w1
    dw2 = (1/m) * np.dot(dz, x2) + (l2_lambda/m) * w2
    db = (1/m) * np.sum(dz)
    return dw1, dw2, db

with st.sidebar:
    st.markdown("### Architecture")
    task_type = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)
    
    st.markdown("### Data Synthesis")
    if task_type == "Regression":
        dataset_type = st.selectbox("Distribution", ["Linear"])
    else:
        dataset_type = st.selectbox("Distribution", ["Blobs (Classification)", "Moons (Classification)"])
        
    num_samples = st.slider("Samples", 50, 500, 150, 10)
    noise_level = st.slider("Noise Intensity", 0.1, 3.0, 1.0, 0.1)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Hyperparameters")
    learning_rate = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0], value=0.1)
    l2_lambda = st.slider("L2 Regularization", 0.0, 2.0, 0.0, 0.1)
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 500], value=64)
    epochs_to_run = st.number_input("Epochs per run", 10, 500, 50, 10)
    anim_speed = st.slider("Render Delay (s)", 0.0, 0.2, 0.02, 0.01)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_reset, col_train = st.columns(2)
    with col_reset:
        if st.button("Reset"):
            st.session_state.w1 = np.random.uniform(-2, 2)
            st.session_state.w2 = np.random.uniform(-2, 2)
            st.session_state.b = np.random.uniform(-2, 2)
            st.session_state.loss_history = []
            st.session_state.log_history = []
            st.session_state.epoch_count = 0
            st.rerun()
    with col_train:
        train_trigger = st.button("Train")

x1_data, x2_data, y_data = generate_dataset(dataset_type, num_samples, noise_level)

if st.session_state.epoch_count == 0 and len(st.session_state.loss_history) == 0:
    preds = calculate_predictions(x1_data, x2_data, st.session_state.w1, st.session_state.w2, st.session_state.b, task_type)
    initial_loss = calculate_loss(y_data, preds, task_type, st.session_state.w1, st.session_state.w2, l2_lambda)
    st.session_state.current_mse = initial_loss
    st.session_state.log_history.insert(0, f"<span>SYSTEM INITIALIZED. Target Loss: {initial_loss:.4f}</span>")

st.markdown('<div class="main-title">NeuralLab Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Multivariable Gradient Descent Engine v2.0</div>', unsafe_allow_html=True)

metrics_placeholder = st.empty()
progress_container = st.empty()

tab1, tab2, tab3 = st.tabs(["Visualization Arena", "Analytics & Diagnostics", "Data Inspector"])

with tab1:
    col_main_chart, col_side_logs = st.columns([7, 3])
    
    with col_main_chart:
        chart_placeholder = st.empty()
        
    with col_side_logs:
        loss_chart_placeholder = st.empty()
        logs_placeholder = st.empty()

with tab2:
    col_resid, col_dist = st.columns(2)
    resid_placeholder = st.empty()
    dist_placeholder = st.empty()

with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### Raw Synthetic Dataset")
    df = pd.DataFrame({"Feature X1": x1_data, "Feature X2": x2_data, "Target Y": y_data})
    st.dataframe(df, use_container_width=True, height=400)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Dataset (CSV)", csv, "neurallab_dataset.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard(w1, w2, b, epoch, loss_hist, logs, is_active=False):
    preds = calculate_predictions(x1_data, x2_data, w1, w2, b, task_type)
    current_loss = calculate_loss(y_data, preds, task_type, w1, w2, l2_lambda)
    
    with metrics_placeholder.container():
        st.markdown('<div class="glass-panel" style="padding: 15px;">', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Weight (W1)", f"{w1:.4f}")
        m2.metric("Weight (W2)", f"{w2:.4f}")
        m3.metric("Bias (B)", f"{b:.4f}")
        m4.metric("Loss", f"{current_loss:.4f}", delta=f"{(current_loss - loss_hist[-2]):.4f}" if len(loss_hist) > 1 else None, delta_color="inverse")
        m5.metric("Epoch", f"{epoch}")
        st.markdown('</div>', unsafe_allow_html=True)

    fig_main = go.Figure()

    if task_type == "Regression":
        x1_grid = np.linspace(min(x1_data)-1, max(x1_data)+1, 30)
        x2_grid = np.linspace(min(x2_data)-1, max(x2_data)+1, 30)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
        z_mesh = w1 * x1_mesh + w2 * x2_mesh + b
        
        fig_main.add_trace(go.Scatter3d(
            x=x1_data, y=x2_data, z=y_data,
            mode='markers',
            marker=dict(size=4, color=secondary, line=dict(width=1, color=bg_color)),
            name='Target Data'
        ))
        
        fig_main.add_trace(go.Surface(
            x=x1_grid, y=x2_grid, z=z_mesh,
            colorscale='Viridis', opacity=0.7,
            contours_z=dict(show=True, usecolormap=True, project_z=True),
            name='Prediction Plane'
        ))
        
        cam_rotation = epoch * 0.02 if is_active else 0
        cam_x = 1.8 * np.cos(cam_rotation)
        cam_y = 1.8 * np.sin(cam_rotation)
        
        fig_main.update_layout(
            scene=dict(
                xaxis_title='Feature X1', yaxis_title='Feature X2', zaxis_title='Target Y',
                camera=dict(eye=dict(x=cam_x, y=cam_y, z=0.5)),
                xaxis=dict(gridcolor=grid_color, backgroundcolor="rgba(0,0,0,0)", tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
                yaxis=dict(gridcolor=grid_color, backgroundcolor="rgba(0,0,0,0)", tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
                zaxis=dict(gridcolor=grid_color, backgroundcolor="rgba(0,0,0,0)", tickfont=dict(color=text_color), titlefont=dict(color=text_color))
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=600, showlegend=False
        )
    else:
        x1_grid = np.linspace(min(x1_data)-1, max(x1_data)+1, 100)
        x2_grid = np.linspace(min(x2_data)-1, max(x2_data)+1, 100)
        x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
        z_mesh = sigmoid(w1 * x1_mesh + w2 * x2_mesh + b)
        
        fig_main.add_trace(go.Contour(
            x=x1_grid, y=x2_grid, z=z_mesh,
            colorscale='RdBu', opacity=0.3, showscale=False,
            contours=dict(start=0, end=1, size=0.1)
        ))
        
        fig_main.add_trace(go.Contour(
            x=x1_grid, y=x2_grid, z=z_mesh,
            type='contour', colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
            contours=dict(type='constraint', operation='=', value=0.5),
            line=dict(color=primary, width=4), name='Decision Boundary'
        ))
        
        colors = [secondary if y == 0 else primary for y in y_data]
        fig_main.add_trace(go.Scatter(
            x=x1_data, y=x2_data, mode='markers',
            marker=dict(size=8, color=colors, line=dict(width=1, color=bg_color)),
            name='Data Points'
        ))
        
        fig_main.update_layout(
            xaxis_title='Feature X1', yaxis_title='Feature X2',
            xaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
            yaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=600, showlegend=False
        )

    chart_placeholder.plotly_chart(fig_main, use_container_width=True)

    if len(loss_hist) > 0:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=loss_hist, mode='lines',
            line=dict(color=primary, width=3, shape='spline'),
            fill='tozeroy', fillcolor=f"rgba(128, 128, 128, 0.1)"
        ))
        fig_loss.update_layout(
            title=dict(text="Convergence Curve", font=dict(color=text_color)),
            xaxis=dict(title="Epoch", gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
            yaxis=dict(title="Loss", gridcolor=grid_color, type='log', tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250
        )
        loss_chart_placeholder.plotly_chart(fig_loss, use_container_width=True)

    html_logs = "<br>".join(logs[:18])
    logs_placeholder.markdown(f"""
    <div class="glass-panel" style="margin-bottom:0; padding:15px; height: 330px;">
        <h4 style="margin-top:0; color:{text_color}; font-family:monospace; border-bottom:1px solid {border_color}; padding-bottom:10px;">>_ SYSTEM_LOGS</h4>
        <div class="terminal-window">{html_logs}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with tab2:
        if task_type == "Regression":
            residuals = y_data - preds
            
            fig_resid = px.scatter(x=preds, y=residuals, color_discrete_sequence=[secondary])
            fig_resid.add_hline(y=0, line_dash="dash", line_color=primary)
            fig_resid.update_layout(
                title=dict(text="Residuals vs Predicted", font=dict(color=text_color)),
                xaxis_title="Predicted Y", yaxis_title="Residual Error",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
                yaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color))
            )
            resid_placeholder.plotly_chart(fig_resid, use_container_width=True)
            
            fig_dist = px.histogram(residuals, nbins=30, color_discrete_sequence=[primary])
            fig_dist.update_layout(
                title=dict(text="Error Distribution", font=dict(color=text_color)),
                xaxis_title="Error Value", yaxis_title="Frequency",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
                yaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
                showlegend=False
            )
            dist_placeholder.plotly_chart(fig_dist, use_container_width=True)
        else:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=preds[y_data==0], name='Class 0', marker_color=secondary, opacity=0.7))
            fig_hist.add_trace(go.Histogram(x=preds[y_data==1], name='Class 1', marker_color=primary, opacity=0.7))
            fig_hist.update_layout(
                barmode='overlay', title=dict(text="Prediction Probability Distribution", font=dict(color=text_color)),
                xaxis_title="Predicted Probability", yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
                yaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), titlefont=dict(color=text_color)),
                legend=dict(font=dict(color=text_color))
            )
            resid_placeholder.plotly_chart(fig_hist, use_container_width=True)

if not st.session_state.is_training:
    render_dashboard(st.session_state.w1, st.session_state.w2, st.session_state.b, st.session_state.epoch_count, st.session_state.loss_history, st.session_state.log_history)

if train_trigger:
    st.session_state.is_training = True
    progress_bar = progress_container.progress(0)
    
    total_samples = len(y_data)
    actual_batch_size = min(batch_size, total_samples)
    
    for i in range(epochs_to_run):
        indices = np.random.permutation(total_samples)
        x1_shuffled = x1_data[indices]
        x2_shuffled = x2_data[indices]
        y_shuffled = y_data[indices]
        
        for j in range(0, total_samples, actual_batch_size):
            x1_batch = x1_shuffled[j:j+actual_batch_size]
            x2_batch = x2_shuffled[j:j+actual_batch_size]
            y_batch = y_shuffled[j:j+actual_batch_size]
            
            batch_preds = calculate_predictions(x1_batch, x2_batch, st.session_state.w1, st.session_state.w2, st.session_state.b, task_type)
            dw1, dw2, db = compute_gradients(x1_batch, x2_batch, y_batch, batch_preds, st.session_state.w1, st.session_state.w2, l2_lambda)
            
            st.session_state.w1 -= learning_rate * dw1
            st.session_state.w2 -= learning_rate * dw2
            st.session_state.b -= learning_rate * db
            
        st.session_state.epoch_count += 1
        
        full_preds = calculate_predictions(x1_data, x2_data, st.session_state.w1, st.session_state.w2, st.session_state.b, task_type)
        current_loss = calculate_loss(y_data, full_preds, task_type, st.session_state.w1, st.session_state.w2, l2_lambda)
        st.session_state.loss_history.append(current_loss)
        
        log_entry = f"<span><span style='opacity:0.5;'>[{st.session_state.epoch_count:04d}]</span> LOSS: <span style='font-weight:bold;'>{current_loss:.4f}</span> | dW1: {dw1:.3f} | dW2: {dw2:.3f}</span>"
        st.session_state.log_history.insert(0, log_entry)
        
        render_dashboard(st.session_state.w1, st.session_state.w2, st.session_state.b, st.session_state.epoch_count, st.session_state.loss_history, st.session_state.log_history, is_active=True)
        progress_bar.progress((i + 1) / epochs_to_run)
        
        if anim_speed > 0:
            time.sleep(anim_speed)
            
    st.session_state.is_training = False
    progress_container.empty()
    st.toast(f"Training Complete! Final Loss: {current_loss:.4f}")

with st.sidebar:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Export Model")
    model_data = {
        "architecture": task_type,
        "epochs_trained": st.session_state.epoch_count,
        "weights": {"w1": float(st.session_state.w1), "w2": float(st.session_state.w2), "bias": float(st.session_state.b)},
        "final_loss": float(st.session_state.loss_history[-1]) if st.session_state.loss_history else None,
        "hyperparameters": {"learning_rate": learning_rate, "l2_regularization": l2_lambda, "batch_size": batch_size}
    }
    json_str = json.dumps(model_data, indent=4)
    st.download_button("Download Weights (JSON)", json_str, "neurallab_weights.json", "application/json")