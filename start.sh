#!/bin/bash
# FastAPI on 8002 (internal)
uvicorn api:app --host 0.0.0.0 --port 8002 &

# Streamlit on 8501 (internal)
streamlit run eralyzer.py --server.port 8501 --server.address 0.0.0.0 &

# nginx on Railway's public $PORT — routes /debate+/status to FastAPI, rest to Streamlit
sed "s/PORT_PLACEHOLDER/${PORT:-8080}/" /app/nginx.conf.template > /tmp/nginx.conf
nginx -c /tmp/nginx.conf -g "daemon off;"
