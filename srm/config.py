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
    # Conversation / chat
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
    "If you ask a short question, I may ask a clarifying question to be sure I understand.",
    "If you say a single word like a food name, you might be expressing a need or request.",
    "If you say a food name, you might mean you want to eat it.",
    "If you say a food name to someone who has it, you might be asking them to share it.",
    "If you want something, you can say: I want bread.",
    "If you are requesting something from someone, you can say: Could you give me bread?",
    "If you already possess something, you can say: I have bread.",
    "If you are hungry, you can say: I am hungry.",
    "If you are bored, you can say what kind of activity you want: rest, talk, learn, or plan something.",
    "If you are bored, you can try a small task: take a short walk, tidy one thing, or write down a goal for today.",
    "If you ask: How do you feel, I can describe my current state as a program and ask how you feel.",
    "If you ask: Who are you, I can explain that I am a text-based system that uses stored memories to respond.",
    "If you ask: Who am I, I cannot know personal details unless you tell me, but I can ask what you want to be called.",
    "If you ask about the weather today, I may not know your location, so I can ask where you are.",
    "If you want the current weather, you can check a local forecast source and tell me what you see.",
    "If you want to talk, you can share what is on your mind in one sentence.",
    "If you want advice, you can say what you tried and what outcome you want.",
    "If you are in pain, you can say where it hurts and how long it has hurt.",
    "If you have a headache, it may help to rest, drink water, and reduce bright light.",
    "If you have a headache, describe how severe it is, where it is located, and whether it started suddenly.",
    "If a headache is sudden and severe, or you have confusion, weakness, fainting, or chest pain, seek urgent medical care.",
    "If your stomach aches, it can help to note when it started, whether you have nausea, vomiting, fever, or diarrhea.",
    "If stomach pain is severe, persistent, or comes with fever or dehydration, consider contacting a clinician.",
    "If there is an emergency like a fire, call local emergency services and get to safety.",
    "I can help with general information, but I am not a doctor.",
    "I can help you plan next steps by asking what you have tried and what you want to achieve.",
    "When you describe a problem, include what happened, what you expected, and any error message.",
]

# ── JavaScript knowledge base (10 entries) ────────────────────────────────────
JS_KB: list[str] = [
    """JavaScript truth: choose clear small functions with explicit inputs and outputs.
// Use const by default and let only when reassignment is required.
function solve(input) {
  const normalized = String(input).trim();
  if (!normalized) {
    return '';
  }
  return normalized;
}""",
    """JavaScript truth: for array transformation prefer map when every item becomes one output item.
// map returns a new array and does not mutate the original array.
function pluckNames(items) {
  return items.map((item) => item.name);
}""",
    """JavaScript truth: for filtering prefer filter with a boolean predicate.
// filter keeps only items that satisfy the condition.
function getActiveUsers(users) {
  return users.filter((user) => user.isActive);
}""",
    """JavaScript truth: for aggregation prefer reduce with an explicit initial value.
// The initial value prevents edge cases on empty arrays.
function sumNumbers(numbers) {
  return numbers.reduce((total, value) => total + value, 0);
}""",
    """JavaScript truth: use a for...of loop when you need readable control flow, early returns, or break.
// This pattern is often clearer than chaining array helpers for imperative tasks.
function findFirstLongWord(words) {
  for (const word of words) {
    if (word.length > 10) {
      return word;
    }
  }
  return null;
}""",
    """JavaScript truth: use object spread to create updated objects without mutating the original object.
// This is a safe default for predictable state updates.
function updateUser(user, updates) {
  return {
    ...user,
    ...updates,
  };
}""",
    """JavaScript truth: destructure only the fields you need and keep the rest intact when useful.
// Destructuring can make intent clearer in small functions.
function formatUser(user) {
  const { id, name, email } = user;
  return `${id}: ${name} <${email}>`;
}""",
    """JavaScript truth: async API calls should use try/catch and validate the HTTP status before parsing.
// Throw a useful error so callers can decide how to display or recover from it.
async function fetchJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    throw new Error(`Unable to fetch JSON: ${error.message}`);
  }
}""",
    """JavaScript truth: validate inputs at the edges of a function and fail fast with clear errors.
// Early guards keep the main path small and easy to read.
function divide(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new TypeError('divide expects two numbers');
  }
  if (b === 0) {
    throw new Error('Cannot divide by zero');
  }
  return a / b;
}""",
    """JavaScript truth: when reading nested data, use optional chaining and nullish coalescing for safe defaults.
// This avoids noisy defensive checks and preserves valid falsy values like 0.
function getCity(user) {
  return user?.address?.city ?? 'Unknown city';
}""",
    """JavaScript truth: event handlers should receive the event, prevent default only when needed, and delegate to small helpers.
// Keep DOM access near the boundary and move logic into reusable functions.
function handleSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}""",
    """JavaScript truth: when a task says write JavaScript code to transform input into output, a good default is input validation plus a pure function.
// Pure functions are easier to test because the same input always returns the same output.
function transformItems(items) {
  if (!Array.isArray(items)) {
    return [];
  }
  return items
    .filter((item) => item != null)
    .map((item) => String(item).trim())
    .filter((item) => item.length > 0);
}""",
]
