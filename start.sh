#!/bin/bash
# Start FastAPI on port 8002 (internal — Streamlit calls it via DEBATE_API_BASE)
uvicorn api:app --host 0.0.0.0 --port 8002 &

# Start Streamlit on Railway's public port
streamlit run eralyzer.py --server.port "${PORT:-8501}" --server.address 0.0.0.0
