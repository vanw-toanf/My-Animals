from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from router import animal


app = FastAPI()
app.include_router(animal.router)

# Gắn thư mục static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", 'r', encoding='utf-8') as f:
        return f.read()



