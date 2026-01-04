import gc
import json
import torch

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vector_store import VectorStore
from utils import CROSS_ENCODER_MODEL


class RetrievalEvaluator:
    """Evaluates retrieval quality using Soft Precision@K."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required but not found. Please ensure a GPU with CUDA support is available.")

        self.device = torch.device("cuda")
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return

        print(f"[RetrievalEvaluator] Loading {CROSS_ENCODER_MODEL}...")

        self.tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CROSS_ENCODER_MODEL,
            device_map="cuda"
        )

        self.model.eval()
        self._loaded = True

    def unload_model(self):
        if not self._loaded:
            return

        print("[RetrievalEvaluator] Unloading model...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _compute_relevance_scores_batch(self, questions: List[str], documents: List[str]) -> List[float]:
        """
        Compute relevance probabilities for multiple (question, document) pairs.

        Args:
            questions: List of questions
            documents: List of documents (same length as questions)

        Returns:
            List of probabilities (0-1) for each pair.
        """
        if not questions:
            return []

        inputs = self.tokenizer(
            questions,
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)

            # Handle single item case where squeeze removes batch dimension
            if probabilities.dim() == 0:
                return [probabilities.item()]

            return probabilities.tolist()

    def evaluate(
        self,
        vector_store: VectorStore,
        dataset_path: str,
        top_k: int,
        matching_threshold: float,
    ) -> Dict:
        """
        Evaluate retrieval performance on test dataset.

        Args:
            vector_store: VectorStore instance to use for retrieval
            dataset_path: Path to test dataset JSON file
            top_k: Number of documents to retrieve per query
            matching_threshold: Minimum similarity threshold

        Returns:
            Dictionary with evaluation results.
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)

        entries = dataset_data.get("entries", [])
        if not entries:
            return {
                "embedding_model": vector_store.embedding_model_name,
                "metric": vector_store.metric,
                "top_k": top_k,
                "num_test_cases": 0,
                "soft_precision_at_k": 0.0
            }

        questions = [entry["question"] for entry in entries]

        print(f"[RetrievalEvaluator] Batch embedding {len(questions)} questions...")
        embeddings = vector_store.embed_queries(questions)

        print(f"[RetrievalEvaluator] Retrieving documents...")
        all_retrieved_docs = []
        for embedding in embeddings:
            docs = vector_store.retrieve(embedding, top_k, matching_threshold)
            all_retrieved_docs.append(docs)

        # Collect all (question, document) pairs for batch scoring
        all_questions = []
        all_documents = []
        per_question_docs_counts = []

        for question, docs in zip(questions, all_retrieved_docs):
            per_question_docs_counts.append(len(docs))
            for doc in docs:
                all_questions.append(question)
                all_documents.append(doc["text"])

        self._load_model()
        all_scores = []

        if all_questions:
            print(f"[RetrievalEvaluator] Batch scoring {len(all_questions)} (question, document) pairs...")
            all_scores = self._compute_relevance_scores_batch(all_questions, all_documents)

        # Map scores back to per-question precisions
        precisions = []
        score_idx = 0
        for count in per_question_docs_counts:
            if count == 0:
                precisions.append(0.0)
            else:
                question_scores = all_scores[score_idx:score_idx + count]
                precisions.append(sum(question_scores) / len(question_scores))
                score_idx += count

        final_score = sum(precisions) / len(precisions)
        print(f"[RetrievalEvaluator] Soft Precision@{top_k}: {final_score:.4f}")

        return {
            "embedding_model": vector_store.embedding_model_name,
            "metric": vector_store.metric,
            "top_k": top_k,
            "num_test_cases": len(entries),
            "soft_precision_at_k": round(final_score, 4)
        }
