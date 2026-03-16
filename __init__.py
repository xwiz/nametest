"""
srm — Stochastic Resonance Memory

Public API
----------
    from srm import MemoryStore, srm_query

    store = MemoryStore("my.db")
    store.add("The mitochondria is the powerhouse of the cell.")
    result = srm_query("cellular energy", store)
    print(result["response"])
"""

from .store    import MemoryStore
from .pipeline import srm_query
from .config   import SAMPLE_KB

__all__ = ["MemoryStore", "srm_query", "SAMPLE_KB"]
__version__ = "1.1.0"
