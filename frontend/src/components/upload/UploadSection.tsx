import { useState, useRef } from 'react';
import axios from 'axios';
import { Card } from '../layout/Card';
import { uploadPdf } from '../../api/client';
import type { UploadResponse, DatasetInfo } from '../../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface UploadSectionProps {
  onUploadComplete: () => void;
  onError: (error: string) => void;
  datasets: DatasetInfo[];
}

export function UploadSection({ onUploadComplete, onError, datasets }: UploadSectionProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showUploadArea, setShowUploadArea] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const hasDocuments = datasets.length > 0;

  const handleUploadSuccess = () => {
    onUploadComplete();
    setShowUploadArea(false);
  };

  const handleFile = async (file: File) => {
    if (!file.type.includes('pdf')) {
      onError('Please upload a PDF file');
      return;
    }

    setUploading(true);
    setUploadResult(null);

    try {
      const result = await uploadPdf(file);
      setUploadResult(result);
      handleUploadSuccess();
    } catch (err) {
      let message = 'An error occurred while uploading the file';
      if (axios.isAxiosError(err)) {
        message = err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        message = err.message;
      }
      onError(message);
    } finally {
      setUploading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const renderUploadArea = () => (
    <div
      className={`
        relative border-2 border-dashed rounded-lg p-8 text-center transition-colors
        ${dragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-gray-400'}
        ${uploading ? 'opacity-50 pointer-events-none' : 'cursor-pointer'}
      `}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        onChange={handleChange}
        disabled={uploading}
        className="hidden"
      />

      <div className="flex flex-col items-center">
        <svg
          className={`w-12 h-12 mb-4 ${dragActive ? 'text-primary-500' : 'text-gray-400'}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>

        {uploading ? (
          <div className="space-y-2">
            <div className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5 text-primary-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="text-sm font-medium text-gray-700">Processing...</span>
            </div>
            <p className="text-xs text-gray-500">Parsing PDF and generating test questions</p>
          </div>
        ) : (
          <>
            <p className="text-sm font-medium text-gray-700">
              Drop your PDF here, or <span className="text-primary-600">browse</span>
            </p>
            <p className="mt-1 text-xs text-gray-500">PDF files only</p>
          </>
        )}
      </div>
    </div>
  );

  return (
    <Card title="Documents" description="Available documents for evaluation">
      {hasDocuments ? (
        <div className="space-y-4">
          {/* Document list */}
          <div className="space-y-2">
            {datasets.map((dataset) => (
              <div
                key={dataset.pdf_name}
                className="flex items-center justify-between px-4 py-3 bg-gray-50 rounded-lg border border-gray-200"
              >
                <div className="flex items-center gap-3">
                  <svg
                    className="w-5 h-5 text-red-500 shrink-0"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <div>
                    <span className="text-sm font-medium text-gray-700">{dataset.pdf_name}</span>
                    {dataset.has_questions && (
                      <span className="ml-2 text-xs text-gray-500">
                        ({dataset.num_questions} questions)
                      </span>
                    )}
                  </div>
                </div>
                {dataset.has_questions && (
                  <a
                    href={`${API_BASE_URL}/datasets/${encodeURIComponent(dataset.pdf_name)}/questions`}
                    download={`${dataset.pdf_name}-questions.json`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-primary-600 hover:text-primary-700 hover:bg-primary-50 rounded-md transition-colors"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                      />
                    </svg>
                    Download Questions
                  </a>
                )}
              </div>
            ))}
          </div>

          {/* Toggle upload area */}
          {showUploadArea ? (
            <div className="space-y-3">
              <div className="flex justify-end">
                <button
                  onClick={() => setShowUploadArea(false)}
                  className="text-sm text-gray-500 hover:text-gray-700"
                >
                  Cancel
                </button>
              </div>
              {renderUploadArea()}
            </div>
          ) : (
            <button
              onClick={() => setShowUploadArea(true)}
              className="inline-flex items-center gap-2 text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              Add another document
            </button>
          )}
        </div>
      ) : (
        renderUploadArea()
      )}

      {uploadResult && (
        <div className="mt-4 rounded-lg bg-green-50 border border-green-200 p-4">
          <div className="flex items-start">
            <svg className="h-5 w-5 text-green-500 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-green-800">Upload Successful</h3>
              <div className="mt-2 text-sm text-green-700 space-y-1">
                <p><span className="font-medium">File:</span> {uploadResult.filename}</p>
                <p><span className="font-medium">Text extracted:</span> {uploadResult.text_length.toLocaleString()} characters</p>
                <p><span className="font-medium">Questions generated:</span> {uploadResult.questions_generated}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
