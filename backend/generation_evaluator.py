import gc
import os
import json
import torch
import re

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from utils import EVALUATION_MODEL, NLI_MODEL


class GenerationEvaluator:
    """
    Evaluates generation faithfulness using NLI-based claim verification.

    Model faithfulness is measured by:
    1. Parsing the generated answer into atomic claims (using the 32B parameter Qwen model)
    2. For each claim, computing entailment probability against context (using DeBERTa NLI)
    3. Averaging entailment probabilities across all claims
    """

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required but not found. Please ensure a GPU with CUDA support is available.")

        self.device = torch.device("cuda")
        self.evaluation_model = None
        self.evaluation_tokenizer = None
        self.nli_model = None
        self.nli_tokenizer = None
        self._loaded = False

    def load_models(self):
        """Load both evaluation (for claim parsing) and NLI (for entailment) models."""
        if self._loaded:
            return

        hf_token = os.getenv("HF_TOKEN")

        print(f"[GenerationEvaluator] Loading evaluation model: {EVALUATION_MODEL}")
        self.evaluation_tokenizer = AutoTokenizer.from_pretrained(
            EVALUATION_MODEL,
            padding_side='left',
            token=hf_token
        )

        self.evaluation_model = AutoModelForCausalLM.from_pretrained(
            EVALUATION_MODEL,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
            token=hf_token
        )

        self.evaluation_model.eval()

        if self.evaluation_tokenizer.pad_token is None:
            self.evaluation_tokenizer.pad_token = self.evaluation_tokenizer.eos_token

        print(f"[GenerationEvaluator] Evaluation model loaded on {self.device}")

        print(f"[GenerationEvaluator] Loading NLI model: {NLI_MODEL}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL,
            device_map="cuda"
        )
        self.nli_model.eval()

        print(f"[GenerationEvaluator] NLI model loaded on {self.device}")

        self._loaded = True

    def unload_models(self):
        """Unload all models and free GPU memory."""
        if not self._loaded:
            return

        print("[GenerationEvaluator] Unloading models...")

        if self.evaluation_model is not None:
            del self.evaluation_model
            del self.evaluation_tokenizer
            self.evaluation_model = None
            self.evaluation_tokenizer = None

        if self.nli_model is not None:
            del self.nli_model
            del self.nli_tokenizer
            self.nli_model = None
            self.nli_tokenizer = None

        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[GenerationEvaluator] Models unloaded")

    def _parse_claims(self, answer: str) -> List[str]:
        """
        Parse an answer into atomic factual claims using the evaluation model.

        Args:
            answer: The generated text answer

        Returns:
            List of atomic claim strings
        """
        if not answer or not answer.strip():
            return []

        prompt = f"""Extract all factual claims from this answer as a JSON list.
        Rules:
        - A claim is ANY factual statement that can be verified, including short answers like numbers, dates, names, or amounts (e.g., "$2.62", "2025", "John Smith")
        - Each claim should be a single, atomic statement
        - If the answer contains a value or fact, that IS a claim
        - Only return an empty list if the answer explicitly says "I don't know" or "I don't have enough information"

        Answer: {answer}

        Output only a valid JSON list of strings. Examples:
        - Answer: "$2.62" -> ["The value is $2.62"]
        - Answer: "Visa's revenue was $35.9 billion in 2024" -> ["Visa's revenue was $35.9 billion in 2024"]
        - Answer: "I don't know" -> []"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        inputs = self.evaluation_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.evaluation_tokenizer([inputs], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.evaluation_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.evaluation_tokenizer.pad_token_id,
                eos_token_id=self.evaluation_tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.evaluation_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        try:
            # Try to extract the claims array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
                if isinstance(claims, list):
                    return [str(c).strip() for c in claims if c and str(c).strip()]
        except json.JSONDecodeError:
            pass

        # Fallback: treat the whole answer as one claim
        print(f"[GenerationEvaluator] Warning: Could not parse claims from response: {response}")
        return [answer.strip()] if answer.strip() else []

    def _compute_entailment_batch(self, claims: List[str], contexts: List[str]) -> List[float]:
        """
        Compute entailment probabilities for multiple (claim, context) pairs.

        Args:
            claims: List of claims to verify
            contexts: List of contexts (same length as claims)

        Returns:
            List of probabilities (0-1) for each pair
        """
        if not claims:
            return []

        inputs = self.nli_tokenizer(
            contexts,
            claims,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits

            # DeBERTa labels: [entailment, neutral, contradiction]
            probs = torch.softmax(logits, dim=-1)
            entailment_probs = probs[:, 0].tolist()

        return entailment_probs

    def evaluate_faithfulness(self, answer: str, context: str) -> Dict:
        """
        Evaluate the faithfulness of a generated answer.

        Args:
            answer: The generated answer
            context: The retrieved context

        Returns:
            Dictionary with:
                - claims: List of extracted claims
                - claim_scores: Entailment probability for each claim
                - faithfulness: Average entailment probability
        """
        self.load_models()

        claims = self._parse_claims(answer)

        if not claims:
            return {
                "claims": [],
                "claim_scores": [],
                "faithfulness": 0.0
            }

        # Batch compute entailment for all claims
        contexts = [context] * len(claims)
        claim_scores = self._compute_entailment_batch(claims, contexts)

        faithfulness = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0

        return {
            "claims": claims,
            "claim_scores": claim_scores,
            "faithfulness": round(faithfulness, 4)
        }
