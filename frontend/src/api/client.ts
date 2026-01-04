import axios from 'axios';
import type {
  UploadResponse,
  DatasetsResponse,
  EmbeddingsResponse,
  EvalRequest,
  RetrievalEvalResponse,
  GenEvalRequest,
  GenerationEvalResponse,
  CustomEmbeddingRequest,
  CustomEmbeddingResponse,
  ModelsResponse,
  CustomModelRequest,
  CustomModelResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<UploadResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function getDatasets(): Promise<DatasetsResponse> {
  const response = await api.get<DatasetsResponse>('/datasets');
  return response.data;
}

export async function getEmbeddings(): Promise<EmbeddingsResponse> {
  const response = await api.get<EmbeddingsResponse>('/embeddings');
  return response.data;
}

export async function evaluateRetrieval(request: EvalRequest): Promise<RetrievalEvalResponse> {
  const response = await api.post<RetrievalEvalResponse>('/evaluate', request);
  return response.data;
}

export async function evaluateGeneration(request: GenEvalRequest): Promise<GenerationEvalResponse> {
  const response = await api.post<GenerationEvalResponse>('/evaluate-generation', request);
  return response.data;
}

export async function addCustomEmbedding(request: CustomEmbeddingRequest): Promise<CustomEmbeddingResponse> {
  const response = await api.post<CustomEmbeddingResponse>('/embeddings/custom', request);
  return response.data;
}

export async function removeCustomEmbedding(key: string): Promise<{ message: string }> {
  const response = await api.delete<{ message: string }>(`/embeddings/custom/${key}`);
  return response.data;
}

export async function getModels(): Promise<ModelsResponse> {
  const response = await api.get<ModelsResponse>('/models');
  return response.data;
}

export async function addCustomModel(request: CustomModelRequest): Promise<CustomModelResponse> {
  const response = await api.post<CustomModelResponse>('/models/custom', request);
  return response.data;
}

export async function removeCustomModel(key: string): Promise<{ message: string }> {
  const response = await api.delete<{ message: string }>(`/models/custom/${key}`);
  return response.data;
}
