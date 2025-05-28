# minimal_test_app.py
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

st.write("Hello")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("Model loaded")
except Exception as e:
    st.error(e)