import { useState, useEffect } from 'react';
import axios from 'axios';
import { Card } from '../layout/Card';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { evaluateGeneration, getEmbeddings, getModels, addCustomModel, removeCustomModel } from '../../api/client';
import type { GenerationEvalResponse, GenEvalModelResult, EmbeddingInfo, ModelInfo } from '../../types';

type ParamCategory = '<5B' | '5B-10B' | '>10B';

const parseParams = (params: string): number => {
  const match = params.match(/([\d.]+)/);
  if (!match) return 0;
  const value = parseFloat(match[1]);
  if (params.toLowerCase().includes('b')) return value;
  if (params.toLowerCase().includes('m')) return value / 1000;
  return value;
};

const getParamCategory = (params: string): ParamCategory => {
  const value = parseParams(params);
  if (value < 5) return '<5B';
  if (value <= 10) return '5B-10B';
  return '>10B';
};

interface GenerationEvaluationProps {
  pdfNames: string[];
  selectedPdf: string;
  onPdfChange: (pdf: string) => void;
  onError: (error: string) => void;
  onEvaluatingChange: (evaluating: boolean) => void;
}

export function GenerationEvaluation({
  pdfNames,
  selectedPdf,
  onPdfChange,
  onError,
  onEvaluatingChange,
}: GenerationEvaluationProps) {
  const [evaluating, setEvaluating] = useState(false);
  const [topK, setTopK] = useState(3);
  const [questionPct, setQuestionPct] = useState(0.1);
  const [chunkSize, setChunkSize] = useState(256);
  const [result, setResult] = useState<GenerationEvalResponse | null>(null);
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [availableEmbeddings, setAvailableEmbeddings] = useState<EmbeddingInfo[]>([]);
  const [selectedEmbedding, setSelectedEmbedding] = useState('bge-small');
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [customModelId, setCustomModelId] = useState('');
  const [addingCustomModel, setAddingCustomModel] = useState(false);
  const [customModelError, setCustomModelError] = useState('');
  const [evaluationError, setEvaluationError] = useState('');

  const fetchModels = async () => {
    try {
      const response = await getModels();
      setAvailableModels(response.models);
      // Select first model from each category by default
      if (selectedModels.length === 0) {
        const defaultSelected: string[] = [];
        const categories: ParamCategory[] = ['<5B', '5B-10B', '>10B'];
        for (const category of categories) {
          const firstInCategory = response.models.find(
            m => !m.is_custom && getParamCategory(m.params) === category
          );
          if (firstInCategory) {
            defaultSelected.push(firstInCategory.key);
          }
        }
        setSelectedModels(defaultSelected);
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  useEffect(() => {
    const fetchEmbeddings = async () => {
      try {
        const response = await getEmbeddings();
        setAvailableEmbeddings(response.embeddings);
      } catch (err) {
        console.error('Failed to fetch embeddings:', err);
      }
    };
    fetchEmbeddings();
    fetchModels();
  }, []);

  const groupModelsByParams = (models: ModelInfo[]) => {
    const grouped: Record<ParamCategory, ModelInfo[]> = {
      '<5B': [],
      '5B-10B': [],
      '>10B': [],
    };
    models.forEach((model) => {
      grouped[getParamCategory(model.params)].push(model);
    });
    return grouped;
  };

  const handleModelToggle = (key: string) => {
    setSelectedModels((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  const handleSelectAllModels = () => {
    setSelectedModels(availableModels.map(m => m.key));
  };

  const handleDeselectAllModels = () => {
    setSelectedModels([]);
  };

  const handleAddCustomModel = async () => {
    setCustomModelError('');

    if (!customModelId.trim()) {
      setCustomModelError('Please enter a HuggingFace model ID');
      return;
    }

    setAddingCustomModel(true);
    try {
      const result = await addCustomModel({ model_id: customModelId.trim() });
      setCustomModelId('');
      await fetchModels();
      // Auto-select the new model
      setSelectedModels((prev) => [...prev, result.key]);
    } catch (err) {
      let message = 'Failed to add custom model';
      if (axios.isAxiosError(err)) {
        message = err.response?.data?.detail || err.message;
      }
      setCustomModelError(message);
    } finally {
      setAddingCustomModel(false);
    }
  };

  const handleRemoveCustomModel = async (key: string) => {
    try {
      await removeCustomModel(key);
      await fetchModels();
      setSelectedModels((prev) => prev.filter((k) => k !== key));
    } catch (err) {
      let message = 'Failed to remove custom model';
      if (axios.isAxiosError(err)) {
        message = err.response?.data?.detail || err.message;
      }
      onError(message);
    }
  };

  const updateEvaluating = (value: boolean) => {
    setEvaluating(value);
    onEvaluatingChange(value);
  };

  const handleEvaluate = async () => {
    setEvaluationError('');

    if (!selectedPdf) {
      setEvaluationError('Please select a PDF');
      return;
    }

    if (selectedModels.length === 0) {
      setEvaluationError('Please select at least one model');
      return;
    }

    updateEvaluating(true);
    setResult(null);

    try {
      const response = await evaluateGeneration({
        pdf_name: selectedPdf,
        top_k: topK,
        question_percentage: questionPct,
        models: selectedModels,
        embedding: selectedEmbedding,
        chunk_size: chunkSize,
      });
      setResult(response);
    } catch (err) {
      let message = 'An error occurred during generation evaluation';
      if (axios.isAxiosError(err)) {
        message = err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        message = err.message;
      }
      setEvaluationError(message);
    } finally {
      updateEvaluating(false);
    }
  };

  const getBestValues = (results: GenEvalModelResult[]) => {
    return {
      maxFaithfulness: Math.max(...results.map((r) => r.avg_faithfulness)),
      minTtft: Math.min(...results.map((r) => r.avg_ttft)),
      maxTps: Math.max(...results.map((r) => r.avg_tokens_per_second)),
    };
  };

  return (
    <Card
      title="Generation Evaluation"
      description="Evaluate generation faithfulness using NLI-based claim verification"
    >
      <div className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Select
            id="gen-pdf"
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
            id="gen-pct"
            label="Questions"
            value={questionPct}
            onChange={(e) => setQuestionPct(parseFloat(e.target.value))}
            disabled={evaluating}
            options={[
              { value: 0.1, label: '10%' },
              { value: 0.2, label: '20%' },
              { value: 0.3, label: '30%' },
              { value: 0.4, label: '40%' },
              { value: 0.5, label: '50%' },
              { value: 0.6, label: '60%' },
              { value: 0.7, label: '70%' },
              { value: 0.8, label: '80%' },
              { value: 0.9, label: '90%' },
              { value: 1.0, label: '100%' },
            ]}
          />
        </div>

        {/* Retrieval Configuration Section */}
        <div className="border border-gray-200 rounded-lg p-4">
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Retrieval Configuration
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Select
              id="gen-embedding"
              label="Embedding"
              value={selectedEmbedding}
              onChange={(e) => setSelectedEmbedding(e.target.value)}
              disabled={evaluating}
              options={availableEmbeddings.map((e) => ({
                value: e.key,
                label: `${e.model} (${e.dimensions}d)`,
              }))}
            />

            <div>
              <label htmlFor="gen-chunk-size" className="block text-sm font-medium text-gray-700 mb-1">
                Chunk Size
              </label>
              <input
                id="gen-chunk-size"
                type="number"
                min={64}
                max={2048}
                value={chunkSize}
                onChange={(e) => setChunkSize(parseInt(e.target.value) || 256)}
                disabled={evaluating}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
              />
            </div>

            <Select
              id="gen-topk"
              label="Top K"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              disabled={evaluating}
              options={[
                { value: 1, label: '1' },
                { value: 2, label: '2' },
                { value: 3, label: '3' },
                { value: 5, label: '5' },
              ]}
            />
          </div>
        </div>

        {/* Model Selection Section */}
        <div className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <label className="block text-sm font-medium text-gray-700">
              Select Models
            </label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleDeselectAllModels}
                disabled={evaluating}
                className="text-xs text-primary-600 hover:text-primary-800 disabled:opacity-50"
              >
                Deselect All
              </button>
              <span className="text-gray-300">|</span>
              <button
                type="button"
                onClick={handleSelectAllModels}
                disabled={evaluating}
                className="text-xs text-primary-600 hover:text-primary-800 disabled:opacity-50"
              >
                Select All
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {(['<5B', '5B-10B', '>10B'] as const).map((category) => {
              const models = groupModelsByParams(availableModels)[category];
              if (models.length === 0) return null;
              return (
                <div key={category} className="space-y-2">
                  <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    {category === '<5B' ? 'Less than 5B' : category === '5B-10B' ? '5B - 10B' : 'More than 10B'}
                  </h5>
                  <div className="space-y-1">
                    {models.map((model) => (
                      <div key={model.key} className="flex items-center gap-1">
                        <label
                          className={`flex-1 flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                            selectedModels.includes(model.key)
                              ? 'border-primary-500 bg-primary-50 text-primary-700'
                              : 'border-gray-200 bg-white hover:border-gray-300'
                          } ${evaluating ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedModels.includes(model.key)}
                            onChange={() => handleModelToggle(model.key)}
                            disabled={evaluating}
                            className="sr-only"
                          />
                          <a
                            href={`https://huggingface.co/${model.model}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-sm truncate hover:underline"
                          >
                            {model.model}
                          </a>
                          {model.is_custom && (
                            <span className="text-xs px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded">
                              custom
                            </span>
                          )}
                        </label>
                        {model.is_custom && (
                          <button
                            type="button"
                            onClick={() => handleRemoveCustomModel(model.key)}
                            disabled={evaluating}
                            className="p-1.5 text-gray-400 hover:text-red-500 disabled:opacity-50"
                            title="Remove custom model"
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
              );
            })}
          </div>

          {/* Custom Model Section */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center gap-1.5 mb-2">
              <label className="block text-xs font-medium text-gray-600">
                Add Custom Model from HuggingFace
              </label>
              <div className="relative group">
                <span
                  className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-amber-100 text-xs cursor-help text-amber-700 hover:bg-amber-200"
                >
                  !
                </span>
                <div className="absolute left-0 bottom-full mb-2 w-80 p-3 bg-gray-800 text-white text-xs rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                  <p className="font-medium mb-1.5 text-amber-300">Potential issues with custom models:</p>
                  <ul className="list-disc list-inside space-y-1 leading-relaxed">
                    <li>Model may require additional dependencies</li>
                    <li>Model may require a different transformers library version</li>
                    <li>Chat template may not be compatible</li>
                    <li>Some models don't support system prompts</li>
                    <li>VRAM requirements may exceed available memory</li>
                    <li>Generation parameters may need tuning</li>
                  </ul>
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
                  if (customModelError) setCustomModelError('');
                }}
                placeholder="e.g., Qwen/Qwen2.5-7B-Instruct"
                disabled={evaluating || addingCustomModel}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddCustomModel();
                  }
                }}
              />
              <button
                type="button"
                onClick={handleAddCustomModel}
                disabled={evaluating || addingCustomModel || !customModelId.trim()}
                className="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {addingCustomModel ? (
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
            {customModelError && (
              <p className="text-xs text-red-600 mt-1">
                {customModelError}
              </p>
            )}
            <p className="text-xs text-gray-500 mt-1">
              Enter any HuggingFace model ID compatible with transformers
            </p>
          </div>

          <p className="text-xs text-gray-500 mt-3">
            {selectedModels.length} of {availableModels.length} models selected
          </p>
        </div>

        <Button
          onClick={handleEvaluate}
          disabled={evaluating || pdfNames.length === 0 || selectedModels.length === 0}
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
            Evaluating {selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''} (this may take a while)...
          </div>
        )}

        {evaluationError && (
          <div className="mt-4 rounded-lg bg-red-50 border border-red-200 p-4">
            <div className="flex items-start gap-3">
              <svg className="h-5 w-5 text-red-500 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="text-sm font-medium text-red-800">Evaluation Error</h3>
                <p className="mt-1 text-sm text-red-700">{evaluationError}</p>
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="space-y-4 mt-6">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-gray-600">
              <span><span className="font-medium">Embedding:</span> {result.embedding}</span>
              <span><span className="font-medium">Chunk size:</span> {result.chunk_size}</span>
              <span><span className="font-medium">Top K:</span> {result.top_k}</span>
              <span><span className="font-medium">Questions:</span> {result.results[0]?.num_questions || 0}</span>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Model
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Faithfulness
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      TTFT (ms)
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      T/s
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Parameters
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Precision
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      VRAM
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-100">
                  {result.results.map((r) => {
                    const best = getBestValues(result.results);
                    return (
                      <tr key={r.model_key} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3 text-sm">
                          <a
                            href={`https://huggingface.co/${r.model_name}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-primary-600 hover:text-primary-800 hover:underline font-medium"
                          >
                            {r.model_name}
                          </a>
                        </td>
                        <td
                          className={`px-4 py-3 text-sm text-center font-medium ${
                            r.avg_faithfulness === best.maxFaithfulness
                              ? 'bg-green-100 text-green-800'
                              : 'text-gray-700'
                          }`}
                        >
                          {(r.avg_faithfulness * 100).toFixed(2)}%
                        </td>
                        <td
                          className={`px-4 py-3 text-sm text-center font-medium ${
                            r.avg_ttft === best.minTtft
                              ? 'bg-green-100 text-green-800'
                              : 'text-gray-700'
                          }`}
                        >
                          {r.avg_ttft.toFixed(0)}
                        </td>
                        <td
                          className={`px-4 py-3 text-sm text-center font-medium ${
                            r.avg_tokens_per_second === best.maxTps
                              ? 'bg-green-100 text-green-800'
                              : 'text-gray-700'
                          }`}
                        >
                          {r.avg_tokens_per_second.toFixed(1)}
                        </td>
                        <td className="px-4 py-3 text-sm text-center text-gray-700">
                          {r.params}
                        </td>
                        <td className="px-4 py-3 text-sm text-center text-gray-700">
                          {r.precision}
                        </td>
                        <td className="px-4 py-3 text-sm text-center text-gray-700">
                          {r.vram}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="space-y-2">
              {result.results.map((r) => (
                <div key={r.model_key} className="border border-gray-200 rounded-lg overflow-hidden">
                  <button
                    type="button"
                    className="w-full px-4 py-3 text-left flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors"
                    onClick={() => setExpandedModel(expandedModel === r.model_key ? null : r.model_key)}
                  >
                    <span className="text-sm font-medium text-gray-700">
                      {r.model_name} - Question Details
                    </span>
                    <svg
                      className={`w-5 h-5 text-gray-500 transition-transform ${
                        expandedModel === r.model_key ? 'rotate-180' : ''
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {expandedModel === r.model_key && (
                    <div className="p-4 space-y-4 bg-white">
                      {r.details.map((detail, idx) => (
                        <div key={idx} className="border border-gray-100 rounded-lg p-4 space-y-3">
                          <div>
                            <span className="text-xs font-semibold uppercase text-gray-500">Question</span>
                            <p className="text-sm text-gray-800 mt-1">{detail.question}</p>
                          </div>
                          <div>
                            <span className="text-xs font-semibold uppercase text-gray-500">Answer</span>
                            <p className="text-sm text-gray-800 mt-1">{detail.answer}</p>
                          </div>
                          <details className="group">
                            <summary className="cursor-pointer text-xs font-semibold uppercase text-gray-500 hover:text-gray-700">
                              Retrieved Context
                            </summary>
                            <p className="text-xs text-gray-600 mt-2 bg-gray-50 p-2 rounded whitespace-pre-wrap">
                              {detail.context}
                            </p>
                          </details>
                          <div>
                            <span className="text-xs font-semibold uppercase text-gray-500">Claims</span>
                            <ul className="mt-1 space-y-1">
                              {detail.claims.map((claim, cIdx) => (
                                <li key={cIdx} className="text-sm flex items-start gap-2">
                                  <span className="text-gray-700">{claim}</span>
                                  <span
                                    className={`shrink-0 px-1.5 py-0.5 text-xs rounded ${
                                      detail.claim_scores[cIdx] >= 0.5
                                        ? 'bg-green-100 text-green-700'
                                        : 'bg-red-100 text-red-700'
                                    }`}
                                  >
                                    {(detail.claim_scores[cIdx] * 100).toFixed(1)}%
                                  </span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          <div className="flex flex-wrap gap-3 pt-2 border-t border-gray-100">
                            <span
                              className={`px-2 py-1 text-xs font-medium rounded ${
                                detail.faithfulness >= 0.5
                                  ? 'bg-green-100 text-green-700'
                                  : 'bg-red-100 text-red-700'
                              }`}
                            >
                              Faithfulness: {(detail.faithfulness * 100).toFixed(2)}%
                            </span>
                            <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-700 rounded">
                              TTFT: {detail.ttft.toFixed(0)}ms
                            </span>
                            <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-700 rounded">
                              T/s: {detail.tokens_per_second.toFixed(1)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
