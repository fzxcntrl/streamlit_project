import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="NeuralLab | Linear Optimizer")

st.title("🚀 NeuralLab - Linear Optimizer")

st.write("This is your first deployed AIML app!")

# Simple example
x = np.linspace(0, 10, 100)
y = 2 * x + 3

fig = go.Figure()
fig.add_scatter(x=x, y=y, mode='lines', name='y = 2x + 3')

st.plotly_chart(fig)