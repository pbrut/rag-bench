import { useState, useEffect } from 'react';
import { Header } from './components/layout/Header';
import { UploadSection } from './components/upload/UploadSection';
import { RetrievalEvaluation } from './components/evaluation/RetrievalEvaluation';
import { GenerationEvaluation } from './components/evaluation/GenerationEvaluation';
import { ErrorAlert } from './components/ui/ErrorAlert';
import { getDatasets } from './api/client';
import type { DatasetInfo } from './types';

function App() {
  const [error, setError] = useState('');
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [pdfNames, setPdfNames] = useState<string[]>([]);
  const [selectedPdf, setSelectedPdf] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);

  const fetchDatasets = async () => {
    try {
      const result = await getDatasets();
      setDatasets(result.datasets || []);
      setPdfNames(result.pdf_names || []);
      if (result.pdf_names && result.pdf_names.length > 0 && !selectedPdf) {
        setSelectedPdf(result.pdf_names[0]);
      }
    } catch (err) {
      console.error('Failed to fetch datasets:', err);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleError = (message: string) => {
    setError(message);
  };

  const dismissError = () => {
    setError('');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header pauseHealthCheck={isEvaluating} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6">
            <ErrorAlert message={error} onDismiss={dismissError} />
          </div>
        )}

        <div className="space-y-8">
          <UploadSection
            onUploadComplete={fetchDatasets}
            onError={handleError}
            datasets={datasets}
          />

          <RetrievalEvaluation
            pdfNames={pdfNames}
            selectedPdf={selectedPdf}
            onPdfChange={setSelectedPdf}
            onError={handleError}
            onEvaluatingChange={setIsEvaluating}
          />

          <GenerationEvaluation
            pdfNames={pdfNames}
            selectedPdf={selectedPdf}
            onPdfChange={setSelectedPdf}
            onError={handleError}
            onEvaluatingChange={setIsEvaluating}
          />
        </div>
      </main>

      <footer className="border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            RAG Bench - Benchmark and optimize your RAG pipeline configurations
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
