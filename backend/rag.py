import gc
import os
import time
import torch

from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from utils import EVAL_MATCHING_THRESHOLD


class RAG:
    def __init__(self, model_name: str, vector_store):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required but not found. Please ensure a GPU with CUDA support is available.")

        self.device = torch.device("cuda")
        self.model_name = model_name
        self.vector_store = vector_store

        hf_token = os.getenv("HF_TOKEN")

        print(f"Loading tokeniser for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left', token=hf_token)

        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            token=hf_token,
            trust_remote_code=True
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def unload_models(self):
        """Unload model and tokenizer from GPU to free memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 256) -> Dict:
        """Generate answer with performance metrics (TTFT and T/s)."""
        system_prompt = (
            "You are an assistant answering questions using the provided context. "
            "The user has no knowledge of the context other than what you provide them, so be thorough and clear in your answers. "
            "If the answer cannot be found in the context, say: 'I don't have enough information to answer this question.' "
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        if "gemma-2" in self.model_name.lower():
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"}
            ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([inputs], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": 1.2,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer
        }

        def generate_with_exception_handling():
            """This function is needed to capture potential exceptions from custom models"""
            try:
                self.model.generate(**generation_kwargs)
            except Exception as e:
                thread_exception[0] = e
                streamer.end()

        ttft = None
        generated_text = []
        token_count = 0
        thread_exception = [None]
        start_time = time.perf_counter()

        thread = Thread(target=generate_with_exception_handling)
        thread.start()

        for text in streamer:
            if ttft is None:
                ttft = time.perf_counter() - start_time
            generated_text.append(text)
            token_count += 1

        thread.join(timeout=60)
        total_time = time.perf_counter() - start_time

        # Check if thread crashed or timed out
        if thread_exception[0] is not None:
            raise RuntimeError(f"Generation failed for {self.model_name}: {thread_exception[0]}")

        if thread.is_alive():
            raise RuntimeError(f"Generation timed out after 1 minute for {self.model_name}.")

        answer = "".join(generated_text).strip()

        if total_time > 0 and token_count > 0:
            tokens_per_second = token_count / total_time
        else:
            tokens_per_second = 0.0

        return {
            "answer": answer,
            "ttft": round(ttft * 1000, 2) if ttft else 0.0,  # Convert to milliseconds
            "tokens_per_second": round(tokens_per_second, 2),
            "total_tokens": token_count,
            "total_time": round(total_time * 1000, 2)  # Convert to milliseconds
        }

    def query(self, question: str, top_k: int = 2) -> Dict:
        retrieved_docs = self.vector_store.retrieve_by_query(question, top_k, EVAL_MATCHING_THRESHOLD)
        context = "\n\n".join([f"[Start Document {i+1}]\n{doc['text']}\n\n[End Document {i+1}]" for i, doc in enumerate(retrieved_docs)])
        gen_result = self.generate_answer(question, context)

        return {
            "question": question,
            "answer": gen_result["answer"],
            "retrieved_docs": retrieved_docs,
            "context": context,
            "ttft": gen_result["ttft"],
            "tokens_per_second": gen_result["tokens_per_second"],
            "total_tokens": gen_result["total_tokens"],
            "total_time": gen_result["total_time"]
        }
