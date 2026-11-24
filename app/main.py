from __future__ import annotations

import time
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from sentence_transformers import CrossEncoder, SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
BOOKS_PATH = DATA_DIR / "books_clean.parquet"
BM25_PATH = DATA_DIR / "bm25_index"
CHUNK_EMBEDDINGS_PATH = DATA_DIR / "book_chunk_embeddings.npy"
CHUNK_BOOK_IDS_PATH = DATA_DIR / "book_chunk_to_book.npy"

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_K = 50
FUSED_CANDIDATES = 30
RRF_K = 60
LOCAL_URL = "http://127.0.0.1:8000/"


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


class BookSearch:
    """Load the saved retrieval artifacts once and serve hybrid book search."""

    def __init__(self) -> None:
        required_paths = (
            BOOKS_PATH,
            BM25_PATH,
            CHUNK_EMBEDDINGS_PATH,
            CHUNK_BOOK_IDS_PATH,
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Search artifacts are missing. Run the notebook through the "
                "chunked dense section first. Missing: " + ", ".join(missing)
            )

        self.books = pd.read_parquet(BOOKS_PATH).reset_index(drop=True)
        self.books["wiki_id"] = self.books["wiki_id"].astype(str)
        self.bm25 = bm25s.BM25.load(str(BM25_PATH), load_corpus=False, mmap=True)
        self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH, mmap_mode="r")
        self.chunk_to_book = np.load(CHUNK_BOOK_IDS_PATH, mmap_mode="r")

        if len(self.chunk_embeddings) != len(self.chunk_to_book):
            raise ValueError("Chunk embeddings and chunk-to-book map do not align")
        if len(self.chunk_to_book) == 0 or int(self.chunk_to_book.max()) >= len(self.books):
            raise ValueError("Chunk-to-book map contains invalid book row IDs")

        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        self.rerank_texts = (
            self.books[["title", "author", "summary"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .tolist()
        )

    def _book_result(self, row_id: int, score: float, score_type: str) -> dict[str, Any]:
        book = self.books.iloc[row_id]
        return {
            "wiki_id": str(book["wiki_id"]),
            "title": str(book["title"]),
            "author": _optional_text(book["author"]),
            "pub_date": _optional_text(book["pub_date"]),
            "genres": _optional_text(book["genres"]),
            "summary": str(book["summary"])[:360],
            "score": float(score),
            "score_type": score_type,
        }

    def search(self, query: str, k: int) -> dict[str, Any]:
        total_started = time.perf_counter()

        started = time.perf_counter()
        bm25_ids, bm25_scores = self.bm25.retrieve(
            bm25s.tokenize(query), k=RETRIEVE_K
        )
        bm25_ms = (time.perf_counter() - started) * 1000
        bm25_ranked = [
            (int(row_id), float(score))
            for row_id, score in zip(bm25_ids[0], bm25_scores[0])
        ]

        started = time.perf_counter()
        query_embedding = self.dense_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        chunk_scores = self.chunk_embeddings @ query_embedding
        book_scores = np.full(len(self.books), -np.inf, dtype=np.float32)
        np.maximum.at(book_scores, self.chunk_to_book, chunk_scores)
        dense_ids = np.argpartition(book_scores, -RETRIEVE_K)[-RETRIEVE_K:]
        dense_ids = dense_ids[np.argsort(book_scores[dense_ids])[::-1]]
        dense_ranked = [
            (int(row_id), float(book_scores[row_id])) for row_id in dense_ids
        ]
        dense_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        rrf_scores: dict[int, float] = {}
        for ranked_list in (bm25_ranked, dense_ranked):
            for rank, (row_id, _) in enumerate(ranked_list, start=1):
                rrf_scores[row_id] = rrf_scores.get(row_id, 0.0) + 1.0 / (RRF_K + rank)
        fused_ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        fused_candidates = fused_ranked[:FUSED_CANDIDATES]
        fusion_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        pairs = [[query, self.rerank_texts[row_id]] for row_id, _ in fused_candidates]
        reranker_scores = np.asarray(
            self.cross_encoder.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)
        reranker_order = np.argsort(reranker_scores)[::-1]
        reranked_ranked = [
            (fused_candidates[int(i)][0], float(reranker_scores[int(i)]))
            for i in reranker_order
        ]
        reranker_ms = (time.perf_counter() - started) * 1000

        total_ms = (time.perf_counter() - total_started) * 1000
        timings = {
            "bm25": round(bm25_ms, 3),
            "dense_chunked": round(dense_ms, 3),
            "rrf": round(fusion_ms, 3),
            "reranker": round(reranker_ms, 3),
            "total": round(total_ms, 3),
        }

        return {
            "query": query,
            "k": k,
            "rerank_candidate_count": len(fused_candidates),
            "results": {
                "bm25": [self._book_result(i, s, "bm25") for i, s in bm25_ranked[:k]],
                "dense": [
                    self._book_result(i, s, "cosine_similarity")
                    for i, s in dense_ranked[:k]
                ],
                "fused": [
                    self._book_result(i, s, "reciprocal_rank_fusion")
                    for i, s in fused_ranked[:k]
                ],
                "reranked": [
                    self._book_result(i, s, "cross_encoder_logit")
                    for i, s in reranked_ranked[:k]
                ],
            },
            "timings_ms": timings,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.book_search = BookSearch()
    # Give Uvicorn a moment to bind its port before opening the local homepage.
    # Set BOOKFINDER_OPEN_BROWSER=0 to disable this (for example, in CI or a server).
    running_in_space = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE"))
    browser_enabled = os.getenv("BOOKFINDER_OPEN_BROWSER", "1") != "0"
    if browser_enabled and not running_in_space:
        timer = threading.Timer(1.0, webbrowser.open_new_tab, args=(LOCAL_URL,))
        timer.daemon = True
        timer.start()
    yield
    app.state.book_search = None


app = FastAPI(title="Tip of the Tongue Book Finder", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html", media_type="text/html")


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    search_engine: BookSearch | None = getattr(request.app.state, "book_search", None)
    return {
        "status": "ok" if search_engine is not None else "loading",
        "books": len(search_engine.books) if search_engine is not None else 0,
    }


@app.get("/search")
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=FUSED_CANDIDATES),
) -> dict[str, Any]:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank")
    search_engine: BookSearch | None = getattr(request.app.state, "book_search", None)
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search is not ready")
    return search_engine.search(query, k)
