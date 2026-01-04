// Upload types
export interface UploadResponse {
  message: string;
  filename: string;
  text_length: number;
  questions_generated: number;
}

// Dataset types
export interface DatasetInfo {
  pdf_name: string;
  num_questions: number;
  text_length: number;
  has_questions: boolean;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
  pdf_names: string[];
}

// Embedding types
export interface EmbeddingInfo {
  key: string;
  model: string;
  dimensions: number;
  is_custom?: boolean;
}

export interface EmbeddingsResponse {
  embeddings: EmbeddingInfo[];
}

export interface CustomEmbeddingRequest {
  model_id: string;
}

export interface CustomEmbeddingResponse {
  key: string;
  model: string;
  dimensions: number;
  message: string;
}

// Model types for generation
export interface ModelInfo {
  key: string;
  model: string;
  params: string;
  precision: string;
  vram: string;
  is_custom?: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface CustomModelRequest {
  model_id: string;
}

export interface CustomModelResponse {
  key: string;
  model: string;
  params: string;
  precision: string;
  vram: string;
  message: string;
}

// Retrieval Evaluation types
export interface ChunkSizes {
  small: number;
  medium: number;
  large: number;
}

export interface EvalRequest {
  pdf_name: string;
  top_k: number;
  embeddings?: string[];  // List of embedding keys to evaluate
  chunk_sizes?: ChunkSizes;  // Custom chunk sizes for small/medium/large
}

export interface EvalResult {
  embedding_key: string;
  embedding_model: string;
  dimensions: number;
  chunk_config: 'small' | 'medium' | 'large';
  soft_precision_at_k: number;
}

export interface RetrievalEvalResponse {
  pdf_name: string;
  top_k: number;
  results: EvalResult[];
}

// Generation Evaluation types
export interface GenEvalRequest {
  pdf_name: string;
  top_k: number;
  question_percentage: number;
  models?: string[];  // List of model keys to evaluate
  embedding?: string;  // Embedding key to use
  chunk_size?: number;  // Chunk size for retrieval
}

export interface ClaimDetail {
  question: string;
  answer: string;
  context: string;
  claims: string[];
  claim_scores: number[];
  faithfulness: number;
  ttft: number;
  tokens_per_second: number;
}

export interface GenEvalModelResult {
  model_key: string;
  model_name: string;
  params: string;
  precision: string;
  vram: string;
  avg_faithfulness: number;
  avg_ttft: number;
  avg_tokens_per_second: number;
  num_questions: number;
  details: ClaimDetail[];
}

export interface GenerationEvalResponse {
  pdf_name: string;
  embedding: string;
  chunk_size: number;
  top_k: number;
  results: GenEvalModelResult[];
}

