"""
srm/api.py — FastAPI-based REST API for SRM.

Provides programmatic access to SRM functionality via HTTP endpoints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import (
    TOP_K, NUM_CASTS, NOISE,
    MIN_COS, VOTE_FLOOR,
    W_VOTE, W_COS,
    SIM_THRESH, MAX_WORDS,
)
from .store import MemoryStore
from .pipeline import srm_query, srm_query_cast_reconstruct, srm_query_auto
from .meaning import MeaningDB


# ── Pydantic models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"  # "synth", "reconstruct", or "auto"
    top_k: int = TOP_K
    num_casts: int = NUM_CASTS
    noise: float = NOISE
    min_cos: float = MIN_COS
    vote_floor: int = VOTE_FLOOR
    w_vote: float = W_VOTE
    w_cos: float = W_COS
    sim_thresh: float = SIM_THRESH
    max_words: int = MAX_WORDS
    use_meaning: bool = False


class AddMemoryRequest(BaseModel):
    text: str


class AttractorDetail(BaseModel):
    mem_id: int
    mem_idx: int
    votes: int
    text: str
    hamming_dist: int
    similarity: float
    cosine: float
    hybrid_score: float


class QueryResponse(BaseModel):
    query: str
    expanded_query: Optional[str]
    response: str
    confidence: float
    top_attractors: List[tuple[int, int, str]]
    attractor_details: List[AttractorDetail]
    num_memories: int
    num_casts: int
    noise: float
    meaning_enabled: bool
    auto_selected_mode: Optional[str] = None
    auto_scores: Optional[dict[str, float]] = None


class StatsResponse(BaseModel):
    memories: int
    db_path: str
    casts: int
    noise: float
    top_k: int
    mode: str
    meaning_enabled: bool


class MemoryItem(BaseModel):
    id: int
    text: str


class ListResponse(BaseModel):
    memories: List[MemoryItem]
    count: int


# ── Global state ───────────────────────────────────────────────────────────────

store: MemoryStore | None = None
meaning_db: MeaningDB | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize store on startup."""
    global store, meaning_db
    store = MemoryStore()
    # Optionally load meaning DB if available
    try:
        meaning_db = MeaningDB("meaning.db")
    except Exception:
        meaning_db = None
    yield
    # Cleanup on shutdown
    if store:
        store.close()
    if meaning_db:
        meaning_db.close()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SRM API",
    description="Stochastic Resonance Memory - REST API",
    version="1.1.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SRM API",
        "version": "1.1.0",
        "description": "Stochastic Resonance Memory REST API",
        "endpoints": {
            "POST /query": "Submit a query",
            "POST /add": "Add a memory",
            "DELETE /memories/{id}": "Delete a memory",
            "GET /memories": "List all memories",
            "GET /stats": "Get statistics",
            "DELETE /memories": "Clear all memories",
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Submit a query to SRM.
    
    Args:
        req: QueryRequest with query text and optional parameters
    
    Returns:
        QueryResponse with response and metadata
    """
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")
    
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="Memory store is empty")
    
    if req.mode not in ["synth", "reconstruct", "auto"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'synth', 'reconstruct', or 'auto'")
    
    meaning = meaning_db if req.use_meaning else None
    
    try:
        if req.mode == "auto":
            result = srm_query_auto(
                req.query,
                store,
                top_k=req.top_k,
                num_casts=req.num_casts,
                noise=req.noise,
                min_cos=req.min_cos,
                vote_floor=req.vote_floor,
                w_vote=req.w_vote,
                w_cos=req.w_cos,
                sim_thresh=req.sim_thresh,
                max_words=req.max_words,
                meaning_db=meaning,
            )
        elif req.mode == "reconstruct":
            result = srm_query_cast_reconstruct(
                req.query,
                store,
                num_casts=req.num_casts,
                noise=req.noise,
                sim_thresh=req.sim_thresh,
                max_words=req.max_words,
                meaning_db=meaning,
            )
        else:
            result = srm_query(
                req.query,
                store,
                top_k=req.top_k,
                num_casts=req.num_casts,
                noise=req.noise,
                min_cos=req.min_cos,
                vote_floor=req.vote_floor,
                w_vote=req.w_vote,
                w_cos=req.w_cos,
                sim_thresh=req.sim_thresh,
                max_words=req.max_words,
                meaning_db=meaning,
            )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add")
async def add_memory(req: AddMemoryRequest):
    """
    Add a new memory to the store.
    
    Args:
        req: AddMemoryRequest with text to add
    
    Returns:
        Success message with memory count
    """
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")
    
    success = store.add(req.text)
    if not success:
        raise HTTPException(status_code=400, detail="Memory already exists or invalid text")
    
    return {"message": "Memory added", "total_memories": store.count()}


@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: int):
    """
    Delete a memory by ID.
    
    Args:
        mem_id: Memory ID to delete
    
    Returns:
        Success message
    """
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")
    
    success = store.delete(mem_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory with id {mem_id} not found")
    
    return {"message": f"Memory {mem_id} deleted", "total_memories": store.count()}


@app.get("/memories", response_model=ListResponse)
async def list_memories():
    """
    List all memories in the store.
    
    Returns:
        ListResponse with all memories
    """
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")
    
    ids, texts = store.load_all()
    memories = [MemoryItem(id=id_, text=text) for id_, text in zip(ids, texts)]
    
    return ListResponse(memories=memories, count=len(memories))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get store statistics.
    
    Returns:
        StatsResponse with current statistics
    """
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")
    
    return StatsResponse(
        memories=store.count(),
        db_path=store.db_path,
        casts=NUM_CASTS,
        noise=NOISE,
        top_k=TOP_K,
        mode="auto",
        meaning_enabled=meaning_db is not None,
    )


@app.delete("/memories")
async def clear_memories():
    """
    Clear all memories from the store.
    
    Returns:
        Success message
    """
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")
    
    store.clear()
    
    return {"message": "All memories cleared", "total_memories": 0}
