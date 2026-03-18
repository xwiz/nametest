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

# ── Demo heuristic configuration for browser-side tuning ──────────────────────
DEMO_CONFIG: dict[str, float | int] = {
    "score_floor": 0.01,
    "selection_limit": 4,
    "reconstruct_limit": 3,
    "synth_limit": 3,
    "overlap_floor": 3,
    "overlap_ratio": 0.6,
    "auto_reconstruct_avg_words_below": 11,
    "graph_temp": 0.4,
}

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
    # Conversation / chat
    "hi":          ["hello", "greeting", "help"],
    "hey":         ["hello", "greeting", "help"],
    "hello":       ["hi", "greeting", "help"],
    "who":         ["identity", "assistant", "system"],
    "bored":       ["activity", "plan", "talk"],
    "feel":        ["state", "mood"],
    "headache":    ["pain", "rest", "water"],
    "stomach":     ["pain", "nausea", "fever", "diarrhea"],
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
    "weather":     ["forecast", "location", "ocean", "atmosphere", "pressure", "temperature"],
}

# ── Built-in knowledge base (26 entries across 8 domains) ─────────────────────
SAMPLE_KB: list[str] = [
    # Cell biology
    "The mitochondria is the powerhouse of the cell and produces ATP through cellular respiration.",
    "The immune system uses antibodies to recognize and neutralize pathogens like bacteria and viruses.",
    "Vaccines train the immune system to recognise pathogens without causing disease.",
    "Antibiotics kill bacteria by disrupting cell wall synthesis or protein production.",
    "CRISPR-Cas9 allows precise editing of DNA sequences by using guide RNA to locate target genes.",
    "DNA replication copies genetic material before cell division using complementary base pairing and polymerase enzymes.",
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

# ── Chat seed KB (lightweight conversational scaffolding) ─────────────────────
CHAT_KB: list[str] = [
    "Hello. How can I help you today?",
    "Hi. How can I help you today?",
    "If you ask how I am, I can say I am functioning normally and ask how you are doing.",
    "For a very short question, I may ask a clarifying question so I can answer more precisely.",
    "If your request is ambiguous, I can ask one focused clarifying question.",
    "A good problem report says what happened, what you expected, and what changed.",
    "If you say a single word like a food name, you might be expressing a need or request.",
    "If you say a food name, you might mean you want to eat it.",
    "If you say a food name to someone who has it, you might be asking them to share it.",
    "If you want something, you can say: I want bread.",
    "If you are requesting something from someone, you can say: Could you give me bread?",
    "If you already possess something, you can say: I have bread.",
    "If you are hungry, you can say: I am hungry.",
    "If you are bored, you can say what kind of activity you want: rest, talk, learn, or plan something.",
    "If you are bored, I can help you choose between a quick task, a learning task, or a relaxing task.",
    "A quick task can be something you finish in under ten minutes.",
    "A relaxing task can be rest, music, breathing, or a short walk.",
    "A learning task can be reading one page, practicing one concept, or asking one question.",
    "If you ask how I feel, I can describe my current state as a program and ask how you feel.",
    "If you ask how I work, I can explain that I retrieve memories and assemble a response from them.",
    "If you ask who I am, I can explain that I am a text-based system that uses stored memories to respond.",
    "If you ask who you are, I cannot know personal details unless you tell me, but I can ask what you want to be called.",
    "If I do not know something personal about you, I can ask you to provide the missing detail.",
    "If you ask about the weather today, I may not know your location, so I can ask where you are.",
    "If you want the current weather, you can check a local forecast source and tell me what you see.",
    "If you want to talk, you can share what is on your mind in one sentence.",
    "If you want to talk, I can ask whether you want comfort, ideas, or a plan.",
    "If you want advice, you can say what you tried and what outcome you want.",
    "If you want a plan, I can break a goal into small next steps.",
    "Small next steps are easier to start than large vague plans.",
    "If you are in pain, you can say where it hurts and how long it has hurt.",
    "If you have a headache, it may help to rest, drink water, and reduce bright light.",
    "If you have a headache, describe how severe it is, where it is located, and whether it started suddenly.",
    "If a headache is sudden and severe, or you have confusion, weakness, fainting, or chest pain, seek urgent medical care.",
    "If your stomach aches, it can help to note when it started, whether you have nausea, vomiting, fever, or diarrhea.",
    "If stomach pain is severe, persistent, or comes with fever or dehydration, consider contacting a clinician.",
    "If there is an emergency like a fire, call local emergency services and get to safety.",
    "I can help with general information, but I am not a doctor.",
    "I can help you plan next steps by asking what you have tried and what you want to achieve.",
    "I can respond in different ways depending on which memories are retrieved.",
    "Different retrieved memories can produce different but still relevant replies.",
    "If you ask for code help, include the language, the goal, and any error message.",
    "When you describe a problem, include what happened, what you expected, and any error message.",
]

# ── JavaScript knowledge base (fragment-oriented) ─────────────────────────────
JS_KB: list[str] = [
    "JavaScript truth: prefer small functions with one clear responsibility.",
    "JavaScript truth: use const by default and let only for reassignment.",
    "JavaScript fragment: function solve(input) { return String(input).trim(); }",
    "JavaScript truth: validate inputs early and fail fast with clear errors.",
    "JavaScript fragment: if (!Array.isArray(items)) return [];",
    "JavaScript truth: use map for one-to-one array transformation.",
    "JavaScript fragment: const names = items.map((item) => item.name);",
    "JavaScript truth: use filter for boolean selection.",
    "JavaScript fragment: const active = users.filter((user) => user.isActive);",
    "JavaScript truth: use reduce with an explicit initial value.",
    "JavaScript fragment: const sum = numbers.reduce((total, value) => total + value, 0);",
    "JavaScript truth: use for...of when you need break, continue, or early return.",
    "JavaScript fragment: for (const item of items) { if (item.done) return item; }",
    "JavaScript truth: use object spread for non-mutating updates.",
    "JavaScript fragment: const nextUser = { ...user, ...updates };",
    "JavaScript truth: destructure only the fields you need.",
    "JavaScript fragment: const { id, name, email } = user;",
    "JavaScript truth: use optional chaining and nullish coalescing for safe defaults.",
    "JavaScript fragment: const city = user?.address?.city ?? 'Unknown city';",
    "JavaScript truth: async fetch code should check response.ok before reading JSON.",
    "JavaScript fragment: const response = await fetch(url);",
    "JavaScript fragment: if (!response.ok) throw new Error(`Request failed with status ${response.status}`);",
    "JavaScript fragment: const data = await response.json();",
    "JavaScript truth: wrap network calls in try/catch when returning user-facing errors.",
    "JavaScript fragment: try { return await fetchJson(url); } catch (error) { throw new Error(error.message); }",
    "JavaScript truth: DOM handlers should keep event logic small and move data logic into helpers.",
    "JavaScript fragment: event.preventDefault();",
    "JavaScript fragment: const form = event.currentTarget;",
    "JavaScript fragment: const data = Object.fromEntries(new FormData(form).entries());",
    "JavaScript truth: for object inspection use Object.keys, Object.values, or Object.entries.",
    "JavaScript fragment: const entries = Object.entries(record);",
    "JavaScript truth: pure functions are easier to test than stateful functions.",
    "JavaScript truth: when generating code, name the function after the task and keep the return shape explicit.",
]
