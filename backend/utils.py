import os
import json
import re

from huggingface_hub import model_info

DATA_DIR = "data"
CUSTOM_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "custom_embeddings.json")
CUSTOM_MODELS_FILE = os.path.join(DATA_DIR, "custom_models.json")
EVALUATION_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EVAL_METRIC = "cosine"
EVAL_MATCHING_THRESHOLD = 0.75
GENERATION_EVAL_EMBEDDING = "BAAI/bge-small-en-v1.5"
GENERATION_EVAL_CHUNK_SIZE = 256

EMBEDDING_MAP = {
    # 384 dimensions
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "gte-small": "thenlper/gte-small",
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",

    # 768 dimensions
    "bge-base": "BAAI/bge-base-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
    "gte-base": "thenlper/gte-base",
    "mpnet-base": "sentence-transformers/all-mpnet-base-v2",
    "distilroberta": "sentence-transformers/all-distilroberta-v1",

    # 1024 dimensions
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-large": "intfloat/e5-large-v2",
    "gte-large": "thenlper/gte-large",
    "e5-large-multi": "intfloat/multilingual-e5-large",
    "gte-large-v15": "Alibaba-NLP/gte-large-en-v1.5",
}

EMBEDDING_CODES = {
    # 384 dimensions
    "BAAI/bge-small-en-v1.5": "bge-small",
    "intfloat/e5-small-v2": "e5-small",
    "thenlper/gte-small": "gte-small",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6",
    "sentence-transformers/all-MiniLM-L12-v2": "minilm-l12",

    # 768 dimensions
    "sentence-transformers/all-mpnet-base-v2": "mpnet-base",
    "BAAI/bge-base-en-v1.5": "bge-base",
    "intfloat/e5-base-v2": "e5-base",
    "thenlper/gte-base": "gte-base",
    "sentence-transformers/all-distilroberta-v1": "distilroberta",

    # 1024 dimensions
    "BAAI/bge-large-en-v1.5": "bge-large",
    "intfloat/e5-large-v2": "e5-large",
    "thenlper/gte-large": "gte-large",
    "intfloat/multilingual-e5-large": "e5-large-multi",
    "Alibaba-NLP/gte-large-en-v1.5": "gte-large-v15",
}

EMBEDDING_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "intfloat/e5-small-v2": 384,
    "thenlper/gte-small": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "intfloat/e5-base-v2": 768,
    "thenlper/gte-base": 768,
    "sentence-transformers/all-distilroberta-v1": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-large-v2": 1024,
    "thenlper/gte-large": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "Alibaba-NLP/gte-large-en-v1.5": 1024,
}

MODEL_MAP = {
    # <5B
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "gemma2-2b": "google/gemma-2-2b-it",

    # 5B - 10B
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma2": "google/gemma-2-9b-it",

    # >10B
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct-AWQ",
}


def get_model_specs(model_name: str) -> dict:
    """
    Get model parameters and estimated VRAM from Hugging Face Hub.

    Args:
        model_name: The Hugging Face model identifier

    Returns:
        Dictionary with 'params' and 'vram' keys
    """

    try:
        info = model_info(model_name)

        num_params = None
        if info.safetensors:
            num_params = info.safetensors.total

        if num_params is None:
            return {"params": "N/A", "vram": "N/A"}

        if num_params >= 1e9:
            params_str = f"{num_params / 1e9:.1f}B"
        elif num_params >= 1e6:
            params_str = f"{num_params / 1e6:.0f}M"
        else:
            params_str = f"{num_params / 1e3:.0f}K"

        # Determine precision and estimate VRAM based on dtype
        # AWQ/GPTQ (4-bit): ~0.5 bytes per param
        # INT8: ~1 byte per param
        # FP16/BF16: ~2 bytes per param
        # FP32: ~4 bytes per param
        # Add ~20% overhead for activations/KV cache

        model_lower = model_name.lower()
        if "awq" in model_lower or "gptq" in model_lower or "4bit" in model_lower:
            bytes_per_param = 0.5
            precision = "INT4"
        elif "int8" in model_lower or "8bit" in model_lower:
            bytes_per_param = 1.0
            precision = "INT8"
        else:
            bytes_per_param = 2.0
            precision = "FP16/BF16"

        vram_bytes = num_params * bytes_per_param * 1.2  # 20% overhead
        vram_gb = vram_bytes / (1024 ** 3)
        vram_str = f"~{vram_gb:.0f} GB"

        return {"params": params_str, "precision": precision, "vram": vram_str}

    except Exception as e:
        print(f"[get_model_specs] Error fetching specs for {model_name}: {e}")
        return {"params": "N/A", "precision": "N/A", "vram": "N/A"}


# =============================================================================
# Custom Embeddings Management
# =============================================================================

def load_custom_embeddings() -> dict:
    """Load custom embeddings from file."""
    if os.path.exists(CUSTOM_EMBEDDINGS_FILE):
        try:
            with open(CUSTOM_EMBEDDINGS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_custom_embeddings(embeddings: dict):
    """Save custom embeddings to file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CUSTOM_EMBEDDINGS_FILE, 'w') as f:
        json.dump(embeddings, f, indent=2)


def add_custom_embedding(key: str, model: str, dimensions: int):
    """Add a custom embedding to the registry."""
    custom = load_custom_embeddings()
    custom[key] = {"model": model, "dimensions": dimensions}
    save_custom_embeddings(custom)


def remove_custom_embedding(key: str) -> bool:
    """Remove a custom embedding from the registry."""
    custom = load_custom_embeddings()
    if key in custom:
        del custom[key]
        save_custom_embeddings(custom)
        return True
    return False


def get_all_embeddings() -> dict:
    """Get all embeddings (built-in + custom)."""
    all_embeddings = dict(EMBEDDING_MAP)
    custom = load_custom_embeddings()
    for key, data in custom.items():
        all_embeddings[key] = data["model"]
    return all_embeddings


def get_all_embedding_dimensions() -> dict:
    """Get dimensions for all embeddings (built-in + custom)."""
    all_dims = dict(EMBEDDING_DIMENSIONS)
    custom = load_custom_embeddings()
    for key, data in custom.items():
        all_dims[data["model"]] = data["dimensions"]
    return all_dims


# =============================================================================
# Custom Models Management
# =============================================================================

def load_custom_models() -> dict:
    """Load custom models from file."""
    if os.path.exists(CUSTOM_MODELS_FILE):
        try:
            with open(CUSTOM_MODELS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_custom_models(models: dict):
    """Save custom models to file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CUSTOM_MODELS_FILE, 'w') as f:
        json.dump(models, f, indent=2)


def add_custom_model(key: str, model: str, params: str, precision: str, vram: str):
    """Add a custom model to the registry."""
    custom = load_custom_models()
    custom[key] = {
        "model": model,
        "params": params,
        "precision": precision,
        "vram": vram
    }
    save_custom_models(custom)


def remove_custom_model(key: str) -> bool:
    """Remove a custom model from the registry."""
    custom = load_custom_models()
    if key in custom:
        del custom[key]
        save_custom_models(custom)
        return True
    return False


def get_all_models() -> dict:
    """Get all models (built-in + custom)."""
    all_models = dict(MODEL_MAP)
    custom = load_custom_models()
    for key, data in custom.items():
        all_models[key] = data["model"]
    return all_models


def get_custom_model_info(key: str) -> dict | None:
    """Get info for a custom model by key."""
    custom = load_custom_models()
    return custom.get(key)


def abbreviate_pdf_name(pdf_name: str, max_length: int = 8) -> str:
    """
    Create a short abbreviation from a PDF name.
    E.g., "Visa-2025-revenue" -> "v2025rev" or "visa2025"
    """
    name = pdf_name.replace(".pdf", "").replace(".PDF", "")
    parts = re.split(r'[-_\s]+', name)

    abbrev = ""
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            abbrev += part[:4]
        else:
            abbrev += part[:4].lower()

    abbrev = re.sub(r'[^a-z0-9]', '', abbrev.lower())[:max_length]

    return abbrev if abbrev else "doc"


# =============================================================================
# Chunk and Text File Management
# =============================================================================

def get_chunks_path(filename: str, chunk_size: int = None) -> str:
    """Get path for storing PDF chunks."""
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    if chunk_size:
        return os.path.join(DATA_DIR, f"chunks_{base_name}_{chunk_size}.json")
    return os.path.join(DATA_DIR, f"chunks_{base_name}.json")


def get_raw_text_path(filename: str) -> str:
    """Get path for storing raw PDF text."""
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    return os.path.join(DATA_DIR, f"raw_text_{base_name}.txt")


def save_raw_text(filename: str, text: str):
    """Save raw extracted text for later use with custom chunk sizes."""
    os.makedirs(DATA_DIR, exist_ok=True)
    text_path = get_raw_text_path(filename)
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved raw text ({len(text)} chars) to {text_path}")


def load_raw_text(filename: str) -> str:
    """Load raw extracted text."""
    text_path = get_raw_text_path(filename)
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Raw text file not found: {text_path}")
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_chunks(filename: str, chunks: list, chunk_size: int, chunk_overlap: int):
    """Save chunks to a file for later use during evaluation."""
    os.makedirs(DATA_DIR, exist_ok=True)
    chunks_path = get_chunks_path(filename, chunk_size)
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump({
            "filename": filename,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunks": chunks
        }, f)
    print(f"Saved {len(chunks)} chunks (size={chunk_size}) to {chunks_path}")


def load_existing_chunks(filename: str, chunk_size: int) -> dict | None:
    """Load existing chunks if they exist."""
    chunks_path = get_chunks_path(filename, chunk_size)
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data['chunks'])} existing chunks (size={chunk_size}) from {chunks_path}")
            return data
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Error loading chunks: {e}")
    return None


def get_questions_path(filename: str) -> str:
    """Get path for storing generated questions."""
    os.makedirs(DATA_DIR, exist_ok=True)
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    return os.path.join(DATA_DIR, f"{base_name}-questions.json")


def validate_pdf_files(pdf_name: str) -> tuple[str, str]:
    """ Validate that required PDF files exist. """
    filename = f"{pdf_name}.pdf"
    questions_path = get_questions_path(filename)
    raw_text_path = get_raw_text_path(filename)

    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"No questions found for PDF: {pdf_name}. Please upload the PDF first.")
    if not os.path.exists(raw_text_path):
        raise FileNotFoundError(f"Raw text not found for PDF: {pdf_name}. Please re-upload the PDF.")

    return questions_path, raw_text_path


def get_or_create_chunks(filename: str, chunk_size: int) -> list[str]:
    """Load existing chunks or create new ones."""
    from pdf_processor import PDFProcessor

    existing = load_existing_chunks(filename, chunk_size)
    if existing and existing.get("chunk_size") == chunk_size:
        return existing["chunks"]

    raw_text = load_raw_text(filename)
    overlap = max(20, chunk_size // 5)
    chunks = PDFProcessor.chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    save_chunks(filename, chunks, chunk_size, overlap)
    print(f"Generated and saved {len(chunks)} chunks (size={chunk_size})")
    return chunks