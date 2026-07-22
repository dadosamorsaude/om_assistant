"""Ponto de entrada do servidor da API (redireciona para app.api.server).

Garante suporte a uvicorn api:app e python api.py para compatibilidade com o Dockerfile / Render.
"""
import os
import uvicorn
from app.api.server import app, extract_markdown_from_partial_json

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=port, reload=False)
