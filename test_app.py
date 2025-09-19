#!/usr/bin/env python3
"""
Minimal test app to verify Railway deployment works
"""

from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Test API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello World!", "status": "working"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
