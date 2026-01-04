import { useState, useEffect } from 'react';
import axios from 'axios';
import { Card } from '../layout/Card';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { evaluateRetrieval, getEmbeddings, addCustomEmbedding, removeCustomEmbedding } from '../../api/client';
import type { RetrievalEvalResponse, EvalResult, EmbeddingInfo, ChunkSizes } from '../../types';

const DEFAULT_EMBEDDINGS = ['bge-small', 'bge-base', 'bge-large'];
const DEFAULT_CHUNK_SIZES: ChunkSizes = { small: 128, medium: 256, large: 512 };

interface RetrievalEvaluationProps {
  pdfNames: string[];
  selectedPdf: string;
  onPdfChange: (pdf: string) => void;
  onError: (error: string) => void;
  onEvaluatingChange: (evaluating: boolean) => void;
}

export function RetrievalEvaluation({
  pdfNames,
  selectedPdf,
  onPdfChange,
  onError,
  onEvaluatingChange,
}: RetrievalEvaluationProps) {
  const [evaluating, setEvaluating] = useState(false);
  const [topK, setTopK] = useState(3);
  const [result, setResult] = useState<RetrievalEvalResponse | null>(null);
  const [availableEmbeddings, setAvailableEmbeddings] = useState<EmbeddingInfo[]>([]);
  const [selectedEmbeddings, setSelectedEmbeddings] = useState<string[]>(DEFAULT_EMBEDDINGS);
  const [chunkSizes, setChunkSizes] = useState<ChunkSizes>(DEFAULT_CHUNK_SIZES);
  const [customModelId, setCustomModelId] = useState('');
  const [addingCustom, setAddingCustom] = useState(false);
  const [customEmbeddingError, setCustomEmbeddingError] = useState('');

  const fetchEmbeddings = async () => {
    try {
      const response = await getEmbeddings();
      setAvailableEmbeddings(response.embeddings);
    } catch (err) {
      console.error('Failed to fetch embeddings:', err);
    }
  };

  useEffect(() => {
    fetchEmbeddings();
  }, []);

  const updateEvaluating = (value: boolean) => {
    setEvaluating(value);
    onEvaluatingChange(value);
  };

  const handleEmbeddingToggle = (key: string) => {
    setSelectedEmbeddings((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  const handleSelectAll = () => {
    setSelectedEmbeddings(availableEmbeddings.map((e) => e.key));
  };

  const handleSelectDefaults = () => {
    setSelectedEmbeddings(DEFAULT_EMBEDDINGS);
  };

  const handleAddCustomEmbedding = async () => {
    setCustomEmbeddingError('');

    if (!customModelId.trim()) {
      setCustomEmbeddingError('Please enter a HuggingFace model ID');
      return;
    }

    setAddingCustom(true);
    try {
      const result = await addCustomEmbedding({ model_id: customModelId.trim() });
      setCustomModelId('');
      await fetchEmbeddings();
      // Auto-select the new embedding
      setSelectedEmbeddings((prev) => [...prev, result.key]);
    } catch (err) {
      let message = 'Failed to add custom embedding';
      if (axios.isAxiosError(err)) {
        message = err.response?.data?.detail || err.message;
      }
      setCustomEmbeddingError(message);
    } finally {
      setAddingCustom(false);
    }
  };

  const handleRemoveCustomEmbedding = async (key: string) => {
    try {
      await removeCustomEmbedding(key);
      await fetchEmbeddings();
      // Remove from selection if selected
      setSelectedEmbeddings((prev) => prev.filter((k) => k !== key));
    } catch (err) {
      let message = 'Failed to remove custom embedding';
      if (axios.isAxiosError(err)) {
        message = err.response?.data?.detail || err.message;
      }
      onError(message);
    }
  };

  const handleEvaluate = async () => {
    if (!selectedPdf) {
      onError('Please select a PDF');
      return;
    }

    if (selectedEmbeddings.length === 0) {
      onError('Please select at least one embedding');
      return;
    }

    updateEvaluating(true);
    setResult(null);

    try {
      const response = await evaluateRetrieval({
        pdf_name: selectedPdf,
        top_k: topK,
        embeddings: selectedEmbeddings,
        chunk_sizes: chunkSizes,
      });
      setResult(response);
    } catch (err) {
      let message = 'An error occurred during evaluation';
      if (axios.isAxiosError(err)) {
        message = err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        message = err.message;
      }
      onError(message);
    } finally {
      updateEvaluating(false);
    }
  };

  const groupEmbeddingsByDimension = (embeddings: EmbeddingInfo[]) => {
    const grouped: Record<number, EmbeddingInfo[]> = {};
    embeddings.forEach((e) => {
      if (!grouped[e.dimensions]) {
        grouped[e.dimensions] = [];
      }
      grouped[e.dimensions].push(e);
    });
    return grouped;
  };

  const groupByDimension = (results: EvalResult[]) => {
    const grouped: Record<number, Record<string, { model: string; scores: Record<string, number> }>> = {};

    results.forEach((r) => {
      if (!grouped[r.dimensions]) {
        grouped[r.dimensions] = {};
      }
      if (!grouped[r.dimensions][r.embedding_key]) {
        grouped[r.dimensions][r.embedding_key] = { model: r.embedding_model, scores: {} };
      }
      grouped[r.dimensions][r.embedding_key].scores[r.chunk_config] = r.soft_precision_at_k;
    });

    return grouped;
  };

  return (
    <Card
      title="Retrieval Evaluation"
      description="Evaluate retrieval quality using Soft Precision@K across all configurations"
    >
      <div className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Select
            id="retrieval-pdf"
            label="Select PDF"
            value={selectedPdf}
            onChange={(e) => onPdfChange(e.target.value)}
            disabled={evaluating || pdfNames.length === 0}
            options={
              pdfNames.length === 0
                ? [{ value: '', label: 'No PDFs available' }]
                : pdfNames.map((name) => ({ value: name, label: name }))
            }
          />

          <Select
            id="retrieval-topk"
            label="Top K"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            disabled={evaluating}
            options={[
              { value: 1, label: '1' },
              { value: 2, label: '2' },
              { value: 3, label: '3' },
              { value: 5, label: '5' },
              { value: 10, label: '10' },
            ]}
          />
        </div>

        {/* Chunk Sizes Section */}
        <div className="border border-gray-200 rounded-lg p-4">
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Chunk Sizes
          </label>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label htmlFor="chunk-small" className="block text-xs text-gray-500 mb-1">
                Small
              </label>
              <input
                id="chunk-small"
                type="number"
                min={32}
                max={2048}
                value={chunkSizes.small}
                onChange={(e) => setChunkSizes({ ...chunkSizes, small: parseInt(e.target.value) || 128 })}
                disabled={evaluating}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
              />
            </div>
            <div>
              <label htmlFor="chunk-medium" className="block text-xs text-gray-500 mb-1">
                Medium
              </label>
              <input
                id="chunk-medium"
                type="number"
                min={32}
                max={2048}
                value={chunkSizes.medium}
                onChange={(e) => setChunkSizes({ ...chunkSizes, medium: parseInt(e.target.value) || 256 })}
                disabled={evaluating}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
              />
            </div>
            <div>
              <label htmlFor="chunk-large" className="block text-xs text-gray-500 mb-1">
                Large
              </label>
              <input
                id="chunk-large"
                type="number"
                min={32}
                max={2048}
                value={chunkSizes.large}
                onChange={(e) => setChunkSizes({ ...chunkSizes, large: parseInt(e.target.value) || 512 })}
                disabled={evaluating}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
              />
            </div>
          </div>
        </div>

        {/* Select Embeddings Section */}
        <div className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <label className="block text-sm font-medium text-gray-700">
              Select Embeddings
            </label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleSelectDefaults}
                disabled={evaluating}
                className="text-xs text-primary-600 hover:text-primary-800 disabled:opacity-50"
              >
                Reset to Defaults
              </button>
              <span className="text-gray-300">|</span>
              <button
                type="button"
                onClick={handleSelectAll}
                disabled={evaluating}
                className="text-xs text-primary-600 hover:text-primary-800 disabled:opacity-50"
              >
                Select All
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(groupEmbeddingsByDimension(availableEmbeddings))
              .sort(([a], [b]) => parseInt(a) - parseInt(b))
              .map(([dim, embeddings]) => (
                <div key={dim} className="space-y-2">
                  <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    {dim} Dimensions
                  </h5>
                  <div className="space-y-1">
                    {embeddings.map((embedding) => (
                      <div key={embedding.key} className="flex items-center gap-1">
                        <label
                          className={`flex-1 flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                            selectedEmbeddings.includes(embedding.key)
                              ? 'border-primary-500 bg-primary-50 text-primary-700'
                              : 'border-gray-200 bg-white hover:border-gray-300'
                          } ${evaluating ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedEmbeddings.includes(embedding.key)}
                            onChange={() => handleEmbeddingToggle(embedding.key)}
                            disabled={evaluating}
                            className="sr-only"
                          />
                          <a
                            href={`https://huggingface.co/${embedding.model}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-sm truncate hover:underline"
                          >
                            {embedding.model}
                          </a>
                          {embedding.is_custom && (
                            <span className="text-xs px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded">
                              custom
                            </span>
                          )}
                        </label>
                        {embedding.is_custom && (
                          <button
                            type="button"
                            onClick={() => handleRemoveCustomEmbedding(embedding.key)}
                            disabled={evaluating}
                            className="p-1.5 text-gray-400 hover:text-red-500 disabled:opacity-50"
                            title="Remove custom embedding"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
          </div>

          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center gap-1.5 mb-2">
              <label className="block text-xs font-medium text-gray-600">
                Add Custom Embedding from HuggingFace
              </label>
              <div className="relative group">
                <span
                  className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-gray-200 text-xs cursor-help text-gray-600 hover:bg-gray-300"
                >
                  i
                </span>
                <div className="absolute left-0 bottom-full mb-2 w-80 p-3 bg-gray-800 text-white text-xs rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                  <p className="font-medium mb-1.5">Note on retrieval metrics:</p>
                  <p className="leading-relaxed">
                    All default embeddings are normalised, and the retrieval metric used is cosine similarity.
                    For normalised embeddings, cosine similarity and dot product are mathematically equivalent,
                    and Euclidean distance is monotonically equivalent. If custom embeddings are unnormalised,
                    retrieval scores may differ when using different metrics.
                  </p>
                  <div className="absolute left-4 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={customModelId}
                onChange={(e) => {
                  setCustomModelId(e.target.value);
                  if (customEmbeddingError) setCustomEmbeddingError('');
                }}
                placeholder="e.g., BAAI/bge-m3"
                disabled={evaluating || addingCustom}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddCustomEmbedding();
                  }
                }}
              />
              <button
                type="button"
                onClick={handleAddCustomEmbedding}
                disabled={evaluating || addingCustom || !customModelId.trim()}
                className="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {addingCustom ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Adding...
                  </>
                ) : (
                  'Add'
                )}
              </button>
            </div>
            {customEmbeddingError && (
              <p className="text-xs text-red-600 mt-1">
                {customEmbeddingError}
              </p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Enter any sentence-transformers compatible model ID
            </p>
          </div>

          <p className="text-xs text-gray-500 mt-3">
            {selectedEmbeddings.length} of {availableEmbeddings.length} embeddings selected
          </p>
        </div>

        <Button
          onClick={handleEvaluate}
          disabled={evaluating || pdfNames.length === 0 || selectedEmbeddings.length === 0}
          loading={evaluating}
          className="w-full sm:w-auto"
        >
          {evaluating ? 'Evaluating...' : 'Run Evaluation'}
        </Button>

        {evaluating && (
          <div className="flex items-center justify-center py-8 text-sm text-gray-500">
            <svg className="animate-spin h-5 w-5 mr-3 text-primary-600" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Evaluating {selectedEmbeddings.length} embeddings x 3 chunk sizes...
          </div>
        )}

        {result && (
          <div className="space-y-6 mt-6">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <span className="font-medium">Soft Precision@{result.top_k}</span>
              <span
                className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-gray-200 text-xs cursor-help"
                title="Retrieval metric using cosine similarity. All embeddings are normalized."
              >
                i
              </span>
            </div>

            {Object.entries(groupByDimension(result.results))
              .sort(([a], [b]) => parseInt(a) - parseInt(b))
              .map(([dim, embeddings]) => {
                const allScores = Object.values(embeddings).flatMap((e) =>
                  Object.values(e.scores)
                );
                const maxScore = Math.max(...allScores);

                return (
                  <div key={dim} className="space-y-3">
                    <h4 className="text-sm font-semibold text-gray-800 border-b border-gray-200 pb-2">
                      {dim} Dimensions
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead>
                          <tr className="bg-gray-50">
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                              Embedding
                            </th>
                            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 tracking-wider">
                              CHUNK SIZE {chunkSizes.small}
                            </th>
                            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 tracking-wider">
                              CHUNK SIZE {chunkSizes.medium}
                            </th>
                            <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 tracking-wider">
                              CHUNK SIZE {chunkSizes.large}
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-100">
                          {Object.entries(embeddings).map(([key, data]) => (
                            <tr key={key} className="hover:bg-gray-50 transition-colors">
                              <td className="px-4 py-3 text-sm">
                                <a
                                  href={`https://huggingface.co/${data.model}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-primary-600 hover:text-primary-800 hover:underline font-medium"
                                >
                                  {data.model}
                                </a>
                              </td>
                              {(['small', 'medium', 'large'] as const).map((size) => {
                                const score = data.scores[size];
                                const isBest = score === maxScore;
                                return (
                                  <td
                                    key={size}
                                    className={`px-4 py-3 text-sm text-center font-medium ${
                                      isBest
                                        ? 'bg-green-100 text-green-800'
                                        : 'text-gray-700'
                                    }`}
                                  >
                                    {score !== undefined
                                      ? `${(score * 100).toFixed(2)}%`
                                      : '-'}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </div>
    </Card>
  );
}
