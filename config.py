"""
srm/config.py — all tuneable constants and static data.

Change values here to experiment with SRM behaviour without touching
any algorithmic code.
"""

from __future__ import annotations

# ── SimHash ───────────────────────────────────────────────────────────────────
CODE_BITS: int   = 128          # Hamming-space dimensionality
PACK_BYTES: int  = CODE_BITS // 8

# ── Stochastic traversal ──────────────────────────────────────────────────────
NUM_CASTS: int   = 40           # Probes fired per query
NOISE: float     = 0.12         # Bit-flip probability per cast

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int       = 5            # Candidate attractors to rank
MIN_COS: float   = 0.05         # Minimum cosine to admit an attractor
VOTE_FLOOR: int  = 2            # Minimum votes (alternative to cosine gate)

# ── Synthesis (RWEA) ──────────────────────────────────────────────────────────
SIM_THRESH: float = 0.68        # MMR de-duplication threshold
MAX_WORDS: int    = 120         # Budget for assembled response

# ── Hybrid scoring weights ────────────────────────────────────────────────────
W_VOTE: float    = 0.55         # Weight given to stochastic vote share
W_COS:  float    = 0.45         # Weight given to TF-IDF cosine similarity

# ── Persistence ───────────────────────────────────────────────────────────────
DB_PATH: str     = "srm_memory.db"

# ── NLP ───────────────────────────────────────────────────────────────────────
STOPWORDS: frozenset[str] = frozenset({
    "the","a","an","is","are","was","were","in","of","to","and","or","it","its",
    "this","that","these","those","be","been","by","for","with","on","at","from",
    "as","into","through","during","before","after","above","below","between",
    "each","few","more","most","other","some","such","no","nor","not","only",
    "own","same","so","than","too","very","can","will","just","should","now",
    "have","has","had","do","does","did","would","could","may","might","also",
    "which","who","how","what","when","where","why","about","used","using",
    "their","they","them","we","us","you","he","she","his","her","all","any",
})

# Discourse connectives used during RWEA assembly
CONN_SUPPORT:  list[str] = ["Furthermore, ", "Additionally, ", "In particular, "]
CONN_CONTRAST: list[str] = ["However, ",     "That said, ",    "Notably, "]
CONN_CONCLUDE: list[str] = ["Therefore, ",   "In summary, ",   "Altogether, "]

# ── Vocabulary bridge ─────────────────────────────────────────────────────────
# Surface query terms → domain terms stored in the KB.
# Bridges the vocabulary gap between natural-language queries and the
# specific terminology encoded in memories — without any embeddings.
EXPANSIONS: dict[str, list[str]] = {
    # Computing / ML
    "learn":       ["neural", "train", "weights", "backprop", "gradient"],
    "machine":     ["neural", "network", "algorithm", "learning", "gradient"],
    "machines":    ["neural", "network", "algorithm", "learning"],
    "ai":          ["neural", "network", "learning", "intelligence", "turing"],
    "computer":    ["turing", "algorithm", "neural", "computation"],
    "robot":       ["neural", "algorithm", "reinforcement", "learning"],
    "policy":      ["reinforcement", "reward", "agent", "learning"],
    # Biology / medicine
    "infection":   ["pathogen", "bacteria", "virus", "immune", "antibody"],
    "fight":       ["immune", "antibody", "neutralize", "defend"],
    "sick":        ["pathogen", "bacteria", "virus", "immune", "disease"],
    "disease":     ["pathogen", "bacteria", "virus", "immune", "antibody"],
    "body":        ["immune", "cell", "organism", "biological", "synapse"],
    "brain":       ["neuron", "synapse", "neurotransmitter", "plasticity"],
    "mind":        ["neuron", "synapse", "neurotransmitter", "cognitive"],
    "gene":        ["dna", "rna", "crispr", "chromosome", "genome"],
    "evolution":   ["dna", "gene", "mutation", "selection", "helix"],
    "medicine":    ["antibiotic", "vaccine", "immune", "pathogen"],
    "drug":        ["antibiotic", "bacteria", "cell", "protein"],
    "cell":        ["mitochondria", "atp", "membrane", "nucleus"],
    "energy":      ["atp", "mitochondria", "photosynthesis", "entropy"],
    "plants":      ["photosynthesis", "glucose", "sunlight", "carbon"],
    "food":        ["photosynthesis", "glucose", "sunlight", "carbon"],
    # Physics / cosmology
    "gravity":     ["spacetime", "relativity", "curvature", "einstein"],
    "black":       ["hole", "event", "horizon", "escape", "gravity"],
    "hole":        ["black", "event", "horizon", "gravity", "escape"],
    "space":       ["spacetime", "relativity", "quantum", "cosmos"],
    "universe":    ["spacetime", "relativity", "quantum", "cosmology"],
    "atom":        ["quantum", "particle", "electron", "proton", "higgs"],
    "particle":    ["quantum", "higgs", "boson", "electron", "field"],
    "electricity": ["maxwell", "electromagnetic", "charge", "magnet"],
    "magnet":      ["maxwell", "electromagnetic", "field", "current"],
    "wave":        ["electromagnetic", "light", "maxwell", "quantum"],
    "light":       ["photon", "electromagnetic", "speed", "vacuum"],
    "temperature": ["entropy", "thermodynamics", "heat", "disorder"],
    "disorder":    ["entropy", "thermodynamics", "system"],
    # Earth / environment
    "earthquake":  ["tectonic", "plate", "lithosphere", "seismic"],
    "earthquakes": ["tectonic", "plate", "lithosphere", "seismic"],
    "cause":       ["tectonic", "plate", "lithosphere"],
    "volcano":     ["tectonic", "plate", "lithosphere", "mantle"],
    "ocean":       ["current", "salinity", "temperature", "gradient"],
    "climate":     ["carbon", "atmosphere", "ocean", "cycle"],
    "weather":     ["ocean", "atmosphere", "pressure", "temperature"],
}

# ── Built-in knowledge base (25 entries across 8 domains) ─────────────────────
SAMPLE_KB: list[str] = [
    # Cell biology
    "The mitochondria is the powerhouse of the cell and produces ATP through cellular respiration.",
    "The immune system uses antibodies to recognize and neutralize pathogens like bacteria and viruses.",
    "Vaccines train the immune system to recognise pathogens without causing disease.",
    "Antibiotics kill bacteria by disrupting cell wall synthesis or protein production.",
    "CRISPR-Cas9 allows precise editing of DNA sequences by using guide RNA to locate target genes.",
    "The double helix structure of DNA was discovered by Watson and Crick in 1953.",
    "RNA serves as the intermediary between DNA and protein synthesis through transcription.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plants.",
    "Diffusion allows molecules to move from regions of high concentration to low concentration.",
    # Neuroscience
    "Neurons communicate via electrochemical signals across synapses using neurotransmitters.",
    "Synaptic plasticity underlies learning and memory by strengthening or weakening neural connections.",
    # Cosmology / physics
    "Black holes have gravity so strong that not even light can escape beyond the event horizon.",
    "Einstein's theory of general relativity describes gravity as the curvature of spacetime.",
    "Quantum entanglement links particles so their states remain correlated regardless of distance.",
    "The Higgs boson gives fundamental particles their mass through interaction with the Higgs field.",
    "Maxwell's equations unify electricity and magnetism into a single electromagnetic framework.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "Entropy in thermodynamics measures the degree of disorder in a closed system, always increasing.",
    # AI / computing
    "Neural networks learn by adjusting weights through backpropagation using gradient descent.",
    "Deep reinforcement learning agents learn optimal policies by maximising cumulative reward signals.",
    "The Turing test evaluates machine intelligence by testing whether a human can distinguish it.",
    # Earth science
    "Plate tectonics describes the movement of Earth's lithospheric plates, causing earthquakes.",
    "Ocean currents are driven by temperature gradients, salinity differences, and Earth's rotation.",
    "The carbon cycle moves carbon through the atmosphere, oceans, and living organisms.",
    # History
    "The French Revolution began in 1789 and ultimately led to the rise of Napoleon Bonaparte.",
]
