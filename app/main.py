from fastapi import FastAPI

app = FastAPI(title="Tip of the Tongue Book Finder")


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/search")
def search(q: str, k: int = 10):
    # placeholder: real retrieval comes later
    return {"query": q, "k": k, "results": []}
