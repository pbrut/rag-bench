import os
import time
import json
import glob
import warnings
import torch

from typing import List, Optional
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rag import RAG
from vector_store import VectorStore
from pdf_processor import PDFProcessor
from dataset_generator import generate_test_dataset
from retrieval_evaluator import RetrievalEvaluator
from generation_evaluator import GenerationEvaluator
from utils import *

warnings.filterwarnings("ignore", message=".*AutoAWQ.*deprecated.*", category=DeprecationWarning)

class HealthResponse(BaseModel):
    status: str
    model: str
    documents_count: Optional[int] = None


class UploadResponse(BaseModel):
    message: str
    filename: str
    text_length: int
    questions_generated: int


class ChunkSizes(BaseModel):
    small: int = 128
    medium: int = 256
    large: int = 512


class EvaluateRequest(BaseModel):
    pdf_name: str
    top_k: Optional[int] = 3
    embeddings: Optional[List[str]] = None
    chunk_sizes: Optional[ChunkSizes] = None


class EvaluateResult(BaseModel):
    embedding_key: str
    embedding_model: str
    chunk_config: str
    chunk_size: int
    chunk_overlap: int
    dimensions: int
    top_k: int
    num_test_cases: int
    soft_precision_at_k: float


class EvaluateResponse(BaseModel):
    pdf_name: str
    top_k: int
    results: List[EvaluateResult]


class GenerationEvalRequest(BaseModel):
    pdf_name: str
    top_k: Optional[int] = 3
    question_percentage: Optional[float] = 0.1
    models: Optional[List[str]] = None
    embedding: Optional[str] = None
    chunk_size: Optional[int] = None


class QuestionDetail(BaseModel):
    question: str
    answer: str
    context: str
    claims: List[str]
    claim_scores: List[float]
    faithfulness: float
    ttft: float
    tokens_per_second: float


class ModelResult(BaseModel):
    model_key: str
    model_name: str
    params: str
    precision: str
    vram: str
    avg_faithfulness: float
    avg_ttft: float
    avg_tokens_per_second: float
    num_questions: int
    details: List[QuestionDetail]


class GenerationEvalResponse(BaseModel):
    pdf_name: str
    embedding: str
    chunk_size: int
    top_k: int
    results: List[ModelResult]


class CustomEmbeddingRequest(BaseModel):
    model_id: str


class CustomEmbeddingResponse(BaseModel):
    key: str
    model: str
    dimensions: int
    message: str


class CustomModelRequest(BaseModel):
    model_id: str


class CustomModelResponse(BaseModel):
    key: str
    model: str
    params: str
    precision: str
    vram: str
    message: str


app = FastAPI(
    title="RAG Bench",
    description="A simple RAG benchmarking API for evaluating retrieval and generation performance on PDF documents.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy", "message": "RAG Bench is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document to be processed.
    Extracts text and generates evaluation questions.
    Chunks are generated on-demand during evaluation.

    - **file**: PDF file to upload
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_bytes = await file.read()

        print(f"Parsing PDF: {file.filename}")
        text = PDFProcessor.extract_text_from_pdf(pdf_bytes)
        print(f"Extracted {len(text)} characters")

        save_raw_text(file.filename, text)

        dataset_results = generate_test_dataset(pdf_bytes, file.filename)
        num_questions = dataset_results.get("questions", 0)

        return UploadResponse(
            message=f"PDF processed: {len(text)} characters extracted, {num_questions} questions generated",
            filename=file.filename,
            text_length=len(text),
            questions_generated=num_questions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/datasets")
async def list_datasets():
    """List available datasets by looking for raw_text files and their questions"""
    datasets = []
    pattern = os.path.join(DATA_DIR, "raw_text_*.txt")

    for path in glob.glob(pattern):
        filename = os.path.basename(path)
        pdf_name = filename.replace("raw_text_", "").replace(".txt", "")

        questions_path = get_questions_path(f"{pdf_name}.pdf")
        num_questions = 0
        if os.path.exists(questions_path):
            try:
                with open(questions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                num_questions = len(data.get("entries", []))
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Error reading questions {questions_path}: {e}")

        try:
            text_length = os.path.getsize(path)
        except OSError:
            text_length = 0

        datasets.append({
            "pdf_name": pdf_name,
            "num_questions": num_questions,
            "text_length": text_length,
            "has_questions": num_questions > 0
        })

    datasets.sort(key=lambda x: x["pdf_name"])

    return {
        "datasets": datasets,
        "pdf_names": [d["pdf_name"] for d in datasets]
    }


@app.get("/datasets/{pdf_name}/questions")
async def download_questions(pdf_name: str):
    """Download the questions JSON file for a specific dataset"""
    questions_path = get_questions_path(f"{pdf_name}.pdf")

    if not os.path.exists(questions_path):
        raise HTTPException(status_code=404, detail=f"Questions file not found for {pdf_name}")

    return FileResponse(
        path=questions_path,
        filename=f"{pdf_name}-questions.json",
        media_type="application/json"
    )


@app.get("/embeddings")
async def list_embeddings():
    """List available embedding models for retrieval evaluation (built-in + custom)"""
    all_embeddings = get_all_embeddings()
    all_dimensions = get_all_embedding_dimensions()
    custom_keys = set(load_custom_embeddings().keys())

    embeddings = []
    for key, model in all_embeddings.items():
        dim = all_dimensions.get(model, 0)
        embeddings.append({
            "key": key,
            "model": model,
            "dimensions": dim,
            "is_custom": key in custom_keys
        })

    # Sort by dimensions, then by key
    embeddings.sort(key=lambda x: (x["dimensions"], x["key"]))
    return {"embeddings": embeddings}


@app.post("/embeddings/custom", response_model=CustomEmbeddingResponse)
async def add_custom_embedding(request: CustomEmbeddingRequest):
    """
    Add a custom embedding model from HuggingFace.
    Validates the model and detects its dimensions automatically.

    - **model_id**: HuggingFace model ID (e.g., "BAAI/bge-m3")
    """
    model_id = request.model_id.strip()

    if not model_id:
        raise HTTPException(status_code=400, detail="Model ID cannot be empty")

    if model_id in EMBEDDING_DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is already available as a built-in embedding"
        )

    custom = load_custom_embeddings()
    for key, data in custom.items():
        if data["model"] == model_id:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' is already registered as '{key}'"
            )

    try:
        print(f"Loading model '{model_id}' to detect dimensions...")
        model = SentenceTransformer(model_id)
        test_embedding = model.encode(["test sentence"])
        dimensions = test_embedding.shape[1]

        # Generate a key from the model ID
        # e.g., "BAAI/bge-m3" -> "bge-m3"
        key = model_id.split("/")[-1].lower()
        all_keys = set(EMBEDDING_MAP.keys()) | set(custom.keys())
        base_key = key
        counter = 1
        while key in all_keys:
            key = f"{base_key}-{counter}"
            counter += 1

        add_custom_embedding(key, model_id, int(dimensions))

        del model
        torch.cuda.empty_cache()

        print(f"Added custom embedding: {key} ({model_id}, {dimensions}d)")

        return CustomEmbeddingResponse(
            key=key,
            model=model_id,
            dimensions=int(dimensions),
            message=f"Successfully added '{model_id}' as '{key}' ({dimensions} dimensions)"
        )

    except Exception as e:
        error_msg = str(e)
        if "is not a local folder and is not a valid model identifier" in error_msg:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found on HuggingFace. Please check the model ID and try again."
            )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load model '{model_id}': {error_msg}"
        )


@app.delete("/embeddings/custom/{key}")
async def remove_custom_embedding_endpoint(key: str):
    """Remove a custom embedding model."""
    if key in EMBEDDING_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot remove built-in embedding '{key}'"
        )

    if remove_custom_embedding(key):
        return {"message": f"Successfully removed custom embedding '{key}'"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Custom embedding '{key}' not found"
        )


@app.get("/models")
async def list_models():
    """List available models for generation evaluation (built-in + custom)"""
    custom_models = load_custom_models()

    models = []
    for key, model_name in MODEL_MAP.items():
        specs = get_model_specs(model_name)
        models.append({
            "key": key,
            "model": model_name,
            "params": specs.get("params", "N/A"),
            "precision": specs.get("precision", "N/A"),
            "vram": specs.get("vram", "N/A"),
            "is_custom": False
        })

    for key, data in custom_models.items():
        models.append({
            "key": key,
            "model": data["model"],
            "params": data["params"],
            "precision": data["precision"],
            "vram": data["vram"],
            "is_custom": True
        })

    return {"models": models}


@app.post("/models/custom", response_model=CustomModelResponse)
async def add_custom_model_endpoint(request: CustomModelRequest):
    """
    Add a custom model from HuggingFace.
    Validates the model exists and fetches its specs.
    """
    model_id = request.model_id.strip()

    if not model_id:
        raise HTTPException(status_code=400, detail="Model ID cannot be empty")

    all_models = get_all_models()
    if model_id in all_models.values():
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is already registered"
        )

    key = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    if key in all_models:
        raise HTTPException(
            status_code=400,
            detail=f"A model with key '{key}' already exists"
        )

    # Validate model exists on HuggingFace and get specs
    try:
        specs = get_model_specs(model_id)
        if specs.get("params") == "N/A":
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found on HuggingFace or has no parameter info"
            )

        add_custom_model(
            key=key,
            model=model_id,
            params=specs["params"],
            precision=specs["precision"],
            vram=specs["vram"]
        )

        return CustomModelResponse(
            key=key,
            model=model_id,
            params=specs["params"],
            precision=specs["precision"],
            vram=specs["vram"],
            message=f"Successfully added custom model '{model_id}'"
        )
    
    except Exception as e:
        error_msg = str(e)
        if "is not a local folder and is not a valid model identifier" in error_msg:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found on HuggingFace. Please check the model ID and try again."
            )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to validate model '{model_id}': {error_msg}"
        )


@app.delete("/models/custom/{key}")
async def remove_custom_model_endpoint(key: str):
    """Remove a custom model."""
    if key in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot remove built-in model '{key}'"
        )

    if remove_custom_model(key):
        return {"message": f"Successfully removed custom model '{key}'"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Custom model '{key}' not found"
        )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_retrieval(request: EvaluateRequest):
    """
    Evaluate retrieval performance using Soft Precision@K for all configurations.
    Groups embeddings by dimension (max 5 indices at a time due to Pinecone limit).
    For each dimension group, iterates through all chunk sizes.

    - **pdf_name**: Name of the PDF (e.g., "Visa-2025-revenue")
    - **top_k**: Number of documents to retrieve (default: 3)
    - **chunk_sizes**: Custom chunk sizes for small/medium/large (default: 128/256/512)
    """
    chunk_configs = [
        {"name": "small", "size": 128, "overlap": 25},
        {"name": "medium", "size": 256, "overlap": 50},
        {"name": "large", "size": 512, "overlap": 100},
    ]

    if request.chunk_sizes:
        chunk_configs = [
            {"name": "small", "size": request.chunk_sizes.small, "overlap": max(20, request.chunk_sizes.small // 5)},
            {"name": "medium", "size": request.chunk_sizes.medium, "overlap": max(40, request.chunk_sizes.medium // 5)},
            {"name": "large", "size": request.chunk_sizes.large, "overlap": max(80, request.chunk_sizes.large // 5)},
        ]

    try:
        questions_path, _ = validate_pdf_files(request.pdf_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    original_filename = f"{request.pdf_name}.pdf"

    all_results = []
    all_embeddings = get_all_embeddings()
    all_dimensions = get_all_embedding_dimensions()
    selected_embeddings = all_embeddings

    # Select user specified embeddings
    if request.embeddings:
        selected_embeddings = {k: v for k, v in all_embeddings.items() if k in request.embeddings}

    # Group embeddings by dimension
    embeddings_by_dimension = {}
    for embedding_key, embedding_model in selected_embeddings.items():
        dim = all_dimensions.get(embedding_model, 0)
        if dim not in embeddings_by_dimension:
            embeddings_by_dimension[dim] = {}
        embeddings_by_dimension[dim][embedding_key] = embedding_model

    sorted_dimensions = sorted(embeddings_by_dimension.keys())

    retrieval_evaluator = RetrievalEvaluator()

    for dimension in sorted_dimensions:
        embedding_group = embeddings_by_dimension[dimension]

        print(f"\n{'#'*60}")
        print(f"# Processing {dimension}-dimension embeddings ({len(embedding_group)} models)")
        print(f"{'#'*60}")

        for embedding_key, embedding_model in embedding_group.items():
            print(f"\n{'='*50}")
            print(f"Loading embedding: {embedding_key} ({dimension}d)")
            print(f"{'='*50}")

            try:
                vs = VectorStore(
                    embedding_model_name=embedding_model,
                    metric=EVAL_METRIC,
                    pdf_name=request.pdf_name,
                    init_index=False
                )

                for chunk_config in chunk_configs:
                    config_name = chunk_config["name"]
                    chunk_size = chunk_config["size"]
                    chunk_overlap = chunk_config["overlap"]

                    print(f"\nChunk config: {config_name} (size={chunk_size}, overlap={chunk_overlap})")

                    chunks = get_or_create_chunks(original_filename, chunk_size)

                    vs.init_index(chunk_size)

                    print(f"Adding {len(chunks)} chunks to index...")
                    vs.add_chunks(chunks, original_filename)

                    print(f"Evaluating: {embedding_key} ({dimension}d) + {config_name}")
                    eval_result = retrieval_evaluator.evaluate(
                        vector_store=vs,
                        dataset_path=questions_path,
                        top_k=request.top_k,
                        matching_threshold=EVAL_MATCHING_THRESHOLD,
                    )

                    all_results.append(EvaluateResult(
                        embedding_key=embedding_key,
                        embedding_model=eval_result["embedding_model"],
                        chunk_config=config_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        dimensions=dimension,
                        top_k=eval_result["top_k"],
                        num_test_cases=eval_result["num_test_cases"],
                        soft_precision_at_k=eval_result["soft_precision_at_k"]
                    ))

            except Exception as e:
                print(f"Error evaluating {embedding_key}: {e}")
            finally:
                delete_all_pinecone_indices()
                vs.unload()

    retrieval_evaluator.unload_model()

    if not all_results:
        raise HTTPException(status_code=500, detail="No valid configurations could be evaluated")

    # Sort by embedding_key then chunk_config
    all_results.sort(key=lambda r: (r.embedding_key, r.chunk_config))

    return EvaluateResponse(
        pdf_name=request.pdf_name,
        top_k=request.top_k,
        results=all_results
    )


@app.post("/evaluate-generation", response_model=GenerationEvalResponse)
async def evaluate_generation(request: GenerationEvalRequest):
    """
    Evaluate generation faithfulness for all models in MODEL_MAP.
    Uses NLI-based claim verification to measure if answers are grounded in context.

    - **pdf_name**: Name of the PDF (e.g., "Visa-2025-revenue")
    - **top_k**: Number of documents to retrieve (default: 3)
    - **question_percentage**: Percentage of questions to evaluate, 0.0-1.0 (default: 0.1 = 10%)
    """
    try:
        questions_path, _ = validate_pdf_files(request.pdf_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    with open(questions_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    all_entries = dataset.get("entries", [])

    if not all_entries:
        raise HTTPException(
            status_code=400,
            detail="No test questions available in the dataset"
        )

    total_questions = len(all_entries)
    num_questions = max(1, int(total_questions * request.question_percentage))
    entries = all_entries[:num_questions]

    all_models = get_all_models()
    selected_models = all_models

    # Select user specified models
    if request.models:
        selected_models = {k: v for k, v in all_models.items() if k in request.models}

    all_embeddings = get_all_embeddings()
    embedding_key = request.embedding
    eval_embedding = GENERATION_EVAL_EMBEDDING

    # Select user specified embedding
    if embedding_key and embedding_key in all_embeddings:
        eval_embedding = all_embeddings[embedding_key]

    eval_chunk_size = request.chunk_size if request.chunk_size else GENERATION_EVAL_CHUNK_SIZE

    print(f"\n{'#'*60}")
    print(f"# Generation Evaluation: {request.pdf_name}")
    print(f"# Embedding: {eval_embedding}")
    print(f"# Chunk size: {eval_chunk_size}")
    print(f"# Questions: {len(entries)}/{total_questions} ({request.question_percentage*100:.0f}%)")
    print(f"# Models: {list(selected_models.keys())}")
    print(f"{'#'*60}\n")

    print(f"Creating vector store with {eval_embedding}...")
    vs = VectorStore(
        embedding_model_name=eval_embedding,
        metric=EVAL_METRIC,
        pdf_name=request.pdf_name,
        chunk_size=eval_chunk_size
    )

    # Check if index is populated
    index_stats = vs.index.describe_index_stats()
    total_vectors = index_stats.get('total_vector_count', 0)
    original_filename = f"{request.pdf_name}.pdf"

    if total_vectors > 0:
        print(f"Index already has {total_vectors} vectors, skipping chunk loading.")
    else:
        chunks = get_or_create_chunks(original_filename, eval_chunk_size)
        print(f"Adding {len(chunks)} chunks to index...")
        vs.add_chunks(chunks, original_filename)

    # ==========================================================================
    # Phase 1: Generate answers for all models
    # ==========================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Generating answers for all models")
    print(f"{'='*60}")

    all_generated_responses = {}

    for model_key, model_name in selected_models.items():
        print(f"\n[{model_key}] Loading model: {model_name}")

        try:
            rag = RAG(model_name=model_name, vector_store=vs)
        except Exception as e:
            torch.cuda.empty_cache()
            raise HTTPException(
                status_code=400,
                detail=f"Error loading model '{model_name}': {str(e)}")

        generated_responses = []
        for entry in entries:
            question = entry["question"]

            try:
                result = rag.query(question, top_k=request.top_k)
                generated_responses.append({
                    "question": question,
                    "answer": result["answer"],
                    "context": result["context"],
                    "ttft": result["ttft"],
                    "tokens_per_second": result["tokens_per_second"]
                })
            except Exception as e:
                rag.unload_models()
                raise HTTPException(
                    status_code=400,
                    detail=f"Error generating answer with model '{model_name}': {str(e)}"
                )

        print(f"[{model_key}] Generated {len(generated_responses)} answers, unloading...")
        rag.unload_models()
        all_generated_responses[model_key] = generated_responses

    # Unload vector store embeddings to free VRAM
    vs.unload()

    # ==========================================================================
    # Phase 2: Evaluate faithfulness for all models
    # ==========================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Evaluating faithfulness for all models")
    print(f"{'='*60}")

    gen_evaluator = GenerationEvaluator()
    gen_evaluator.load_models()

    all_results = []

    for model_key, model_name in selected_models.items():
        print(f"\n[{model_key}] Evaluating faithfulness...")

        generated_responses = all_generated_responses[model_key]
        question_details = []

        for resp in generated_responses:
            try:
                eval_result = gen_evaluator.evaluate_faithfulness(
                    resp["answer"],
                    resp["context"]
                )

                question_details.append(QuestionDetail(
                    question=resp["question"],
                    answer=resp["answer"],
                    context=resp["context"],
                    claims=eval_result["claims"],
                    claim_scores=eval_result["claim_scores"],
                    faithfulness=eval_result["faithfulness"],
                    ttft=resp["ttft"],
                    tokens_per_second=resp["tokens_per_second"]
                ))

            except Exception as e:
                print(f"Error evaluating: {e}")
                continue

        avg_faithfulness = 0.0
        avg_ttft = 0.0
        avg_tokens_per_second = 0.0

        if question_details:
            avg_faithfulness = sum(d.faithfulness for d in question_details) / len(question_details)
            avg_ttft = sum(d.ttft for d in question_details) / len(question_details)
            avg_tokens_per_second = sum(d.tokens_per_second for d in question_details) / len(question_details)

        model_spec = get_model_specs(model_name)
        all_results.append(ModelResult(
            model_key=model_key,
            model_name=model_name,
            params=model_spec["params"],
            precision=model_spec["precision"],
            vram=model_spec["vram"],
            avg_faithfulness=round(avg_faithfulness, 4),
            avg_ttft=round(avg_ttft, 2),
            avg_tokens_per_second=round(avg_tokens_per_second, 2),
            num_questions=len(question_details),
            details=question_details
        ))

        print(f"[{model_key}] Faithfulness: {avg_faithfulness:.4f}, TTFT: {avg_ttft:.0f}ms, T/s: {avg_tokens_per_second:.1f}")

    gen_evaluator.unload_models()

    all_results.sort(key=lambda r: r.avg_faithfulness, reverse=True)

    return GenerationEvalResponse(
        pdf_name=request.pdf_name,
        embedding=eval_embedding,
        chunk_size=eval_chunk_size,
        top_k=request.top_k,
        results=all_results
    )


def delete_all_pinecone_indices():
    """Delete all Pinecone indices to free up the 5 index limit. Not needed if using a paid plan."""

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("No Pinecone API key found")

    pc = Pinecone(api_key=pinecone_api_key)

    for index in pc.list_indexes():
        print(f"Deleting index: {index.name}")
        pc.delete_index(index.name)

    # Wait for indices to be deleted
    while list(pc.list_indexes()):
        print("Waiting for indices to be deleted...")
        time.sleep(1)

    print("All indices deleted")
