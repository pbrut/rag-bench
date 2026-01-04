import os
import gc
import time
import torch

from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from utils import EMBEDDING_CODES, get_all_embedding_dimensions, abbreviate_pdf_name


class VectorStore:
    """Vector store using Pinecone and HuggingFace embeddings."""

    def __init__(self, embedding_model_name: str, metric: str, pdf_name: str, chunk_size: int = 0, init_index: bool = True):
        """
        Initialize Pinecone vector store with embeddings.

        Args:
            embedding_model_name: Name of the embedding model
            metric: Distance metric for similarity search
            pdf_name: Name of the PDF document
            chunk_size: Size of text chunks
            init_index: If True, initialize Pinecone index immediately. If False, call init_index() later.
        """
        self.embedding_model_name = embedding_model_name
        self.metric = metric
        self.chunk_size = chunk_size
        self.pdf_name = pdf_name
        self.index = None

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("No Pinecone API key found. Please set the PINECONE_API_KEY environment variable.")

        all_dimensions = get_all_embedding_dimensions()
        if embedding_model_name not in all_dimensions:
            raise ValueError(f"Unknown embedding model: {embedding_model_name}")

        self.dimension = all_dimensions[embedding_model_name]
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"trust_remote_code": True})

        if init_index:
            if chunk_size == 0:
                raise ValueError("chunk_size must be specified when init_index=True")
            self._init_pinecone()

    def unload(self):
        """Unload embeddings model from GPU to free memory."""
        if hasattr(self, 'embeddings') and self.embeddings is not None:
            if hasattr(self.embeddings, '_client'):
                del self.embeddings._client
            del self.embeddings
            self.embeddings = None

        # Force garbage collection before clearing CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            print("[VectorStore] Clearing the embedding model from GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def init_index(self, chunk_size: int):
        """Initialize or switch to a new Pinecone index for the given chunk size."""
        self.chunk_size = chunk_size
        self._init_pinecone()

    def _init_pinecone(self):
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)

            default_embedding_code = self.embedding_model_name.split("/")[-1].replace(".", "-").lower()[:8]
            embedding_code = EMBEDDING_CODES.get(
                self.embedding_model_name,
                default_embedding_code
            )

            if self.pdf_name:
                pdf_abbrev = abbreviate_pdf_name(self.pdf_name)
                index_name = f"{pdf_abbrev}-{embedding_code}-{self.dimension}-{self.chunk_size}"
            else:
                index_name = f"{embedding_code}-{self.dimension}-{self.chunk_size}"

            if len(index_name) > 45:
                raise ValueError(f"Index name '{index_name}' is too long ({len(index_name)} chars). Must be ≤45 chars.")

            if index_name not in [idx.name for idx in pc.list_indexes()]:
                print(f"Creating new Pinecone index: {index_name} with dimension {self.dimension}")
                pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )

                print(f"Waiting for index '{index_name}' to be ready...")
                while not pc.describe_index(index_name).status['ready']:
                    time.sleep(0.5)

            self.index = pc.Index(index_name)
            print(f"Pinecone initialized with index: {index_name}")

        except Exception as e:
            raise RuntimeError(f"Pinecone initialization failed: {e}")

    def add_chunks(self, chunks: list, filename: str, batch_size: int = 100):
        """Add pre-parsed chunks to the index in batches."""
        total_chunks = len(chunks)

        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]

            embeddings = self.embeddings.embed_documents(batch_chunks)

            vectors = [
                (
                    f"{filename}_chunk_{batch_start + i}",
                    embedding,
                    {"text": chunk, "source": filename, "chunk_index": batch_start + i}
                )
                for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings))
            ]

            # Batch upsert to Pinecone
            self.index.upsert(vectors=vectors)

        print(f"Added {total_chunks} chunks from '{filename}'")
        return total_chunks

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Batch embed multiple queries."""
        return self.embeddings.embed_documents(queries)

    def retrieve(self, embedding: List[float], top_k: int, matching_threshold: float) -> List[Dict]:
        """Retrieve relevant documents using a precomputed embedding."""
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )

        return [
            {"text": match.metadata["text"], "score": match.score}
            for match in results.matches if match.score > matching_threshold
        ]

    def retrieve_by_query(self, query: str, top_k: int, matching_threshold: float) -> List[Dict]:
        """Embed query and retrieve relevant documents."""
        embedding = self.embeddings.embed_query(query)
        return self.retrieve(embedding, top_k, matching_threshold)
