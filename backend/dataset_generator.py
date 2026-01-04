import os
import json
import torch

from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from pdf_processor import PDFProcessor
from utils import EVALUATION_MODEL, get_questions_path


def load_existing_questions(filename: str) -> Optional[List[Dict]]:
    """Load existing questions if they exist."""
    questions_path = get_questions_path(filename)
    if os.path.exists(questions_path):
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[DatasetGenerator] Loaded {len(data['entries'])} existing questions from {questions_path}")
            return data['entries']
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"[DatasetGenerator] Error loading questions: {e}")
    return None


def save_questions(filename: str, entries: List[Dict]):
    """Save generated questions to a separate file."""
    questions_path = get_questions_path(filename)
    data = {
        "filename": filename,
        "entries": entries
    }
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[DatasetGenerator] Saved {len(entries)} questions to {questions_path}")


class DatasetGenerator:
    """Generation evaluation questions from an uploaded PDF using Qwen2.5-32B-Instruct-AWQ."""
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

        print(f"[DatasetGenerator] Loading {EVALUATION_MODEL}...")
        hf_token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            EVALUATION_MODEL, padding_side='left', token=hf_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            EVALUATION_MODEL,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
            token=hf_token
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._loaded = True
        print(f"[DatasetGenerator] Model loaded on {self.device}")

    def _unload_model(self):
        if not self._loaded:
            return

        print("[DatasetGenerator] Unloading model and clearing GPU memory...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[DatasetGenerator] GPU memory cleared")

    def _generate_questions_for_chunk(self, chunk_text: str, num_questions: int = 3) -> List[str]:
        """Generate questions that can only be answered by the given text chunk."""

        system_prompt = (
            "You are a dataset generator for RAG evaluation. "
            "Given a text chunk, generate specific questions that can ONLY be answered using information in that chunk. "
            "The questions should be diverse and test different aspects of the content. "
            "Output ONLY the questions, one per line, without numbering or bullet points."
        )

        user_prompt = f"""Generate exactly {num_questions} questions that can only be answered using the following text chunk.
        The questions should:
        - Be specific and answerable from the chunk
        - Cover different facts/aspects mentioned in the chunk
        - Not require external knowledge

        Text chunk:
        \"\"\"
        {chunk_text}
        \"\"\"

        Output {num_questions} questions, one per line:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([inputs], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse questions from response (one per line)
        questions = [q.strip() for q in response.split('\n') if q.strip()]

        # Remove any numbering like "1.", "1)", "-", "*"
        cleaned_questions = []
        for q in questions:
            q = q.lstrip('0123456789.-)*• ').strip()
            if q and q.endswith('?'):
                cleaned_questions.append(q)

        return cleaned_questions[:num_questions]
    
    def generate_questions_from_pdf(self, pdf_bytes: bytes, filename: str, num_questions_per_chunk: int = 3) -> List[Dict]:
        """
        Generate questions from a PDF file.

        Args:
            pdf_bytes: Raw PDF bytes
            filename: Name of the PDF file
            num_questions_per_chunk: Number of questions to generate per chunk

        Returns:
            List of questions
        """
        self._load_model()

        print(f"[DatasetGenerator] Processing PDF: {filename}")
        text = PDFProcessor.extract_text_from_pdf(pdf_bytes)
        chunks = PDFProcessor.chunk_text(text)
        print(f"[DatasetGenerator] Created {len(chunks)} chunks")

        entries = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            print(f"[DatasetGenerator] Generating questions for chunk {i+1}/{len(chunks)}")

            questions = self._generate_questions_for_chunk(
                chunk,
                num_questions=num_questions_per_chunk
            )

            for question in questions:
                entries.append({
                    "question": question,
                    "source_chunk_id": chunk_id,
                    "source_text": chunk
                })
                print(f"  - {question}")

        return entries


def generate_test_dataset(pdf_bytes: bytes, filename: str) -> Dict[str, int]:
    """Generate test dataset from PDF bytes, reusing existing questions if available."""
    entries = load_existing_questions(filename)

    if entries is None:
        generator = DatasetGenerator()
        try:
            entries = generator.generate_questions_from_pdf(pdf_bytes, filename)
            save_questions(filename, entries)
        finally:
            generator._unload_model()

    return {"questions": len(entries)}
