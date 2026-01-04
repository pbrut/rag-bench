import { useState, useEffect, useRef } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface HeaderProps {
  pauseHealthCheck?: boolean;
}

export function Header({ pauseHealthCheck = false }: HeaderProps) {
  const [status, setStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const lastKnownStatus = useRef<'connected' | 'disconnected'>('connected');

  useEffect(() => {
    if (pauseHealthCheck) {
      return;
    }

    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000),
        });
        const newStatus = response.ok ? 'connected' : 'disconnected';
        setStatus(newStatus);
        lastKnownStatus.current = newStatus;
      } catch {
        setStatus('disconnected');
        lastKnownStatus.current = 'disconnected';
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);

    return () => clearInterval(interval);
  }, [pauseHealthCheck]);

  const statusConfig = {
    checking: {
      bg: 'bg-yellow-100',
      text: 'text-yellow-800',
      dot: 'bg-yellow-500',
      label: 'Checking...',
    },
    connected: {
      bg: 'bg-green-100',
      text: 'text-green-800',
      dot: 'bg-green-500',
      label: 'Connected',
    },
    disconnected: {
      bg: 'bg-red-100',
      text: 'text-red-800',
      dot: 'bg-red-500',
      label: 'Disconnected',
    },
  };

  const config = statusConfig[status];

  return (
    <header className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
              RAG Bench
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Benchmark and optimize your RAG pipeline configurations
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.bg} ${config.text}`}>
              <span className={`w-1.5 h-1.5 mr-1.5 rounded-full ${config.dot} ${status === 'connected' ? 'animate-pulse' : ''}`}></span>
              {config.label}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
