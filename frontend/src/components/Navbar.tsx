import React from 'react';
import { RefreshCw } from 'lucide-react';

interface NavbarProps {
  onReset: () => void;
  hasResult: boolean;
  backendStatus: 'checking' | 'online' | 'offline';
}

export const Navbar: React.FC<NavbarProps> = ({ onReset, hasResult, backendStatus }) => {
  const statusLabel = backendStatus === 'checking' ? 'Checking service' : backendStatus === 'online' ? 'Service ready' : 'Service unavailable';
  return (
    <header className="border-b border-border bg-surface sticky top-0 z-50">
      <div className="max-w-7xl mx-auto min-h-14 sm:min-h-16 px-3 sm:px-6 lg:px-8 py-1.5 flex items-center justify-between gap-2">
        {/* Brand */}
        <div className="flex min-w-0 flex-1 items-center">
          <img src="/logo.png" alt="Grain Quality Analyzer" className="h-12 sm:h-14 md:h-16 w-auto max-w-full object-contain object-left" />
        </div>

        {/* Status & Actions */}
        <div className="flex shrink-0 items-center gap-2 sm:gap-4">
          <div role="status" aria-label={statusLabel} title={statusLabel} className="flex items-center gap-2 text-xs text-text-secondary bg-surface-subtle px-2.5 py-2 sm:px-3 rounded-lg border border-border">
            <span className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-success' : backendStatus === 'offline' ? 'bg-error' : 'bg-warning animate-pulse'}`} />
            <span className="hidden sm:inline">{backendStatus === 'checking' ? 'Connecting' : backendStatus === 'online' ? 'Ready' : 'Unavailable'}</span>
          </div>

          {hasResult && (
            <button
              onClick={onReset}
              aria-label="Start a new analysis"
              className="flex min-h-10 items-center gap-1.5 text-xs font-medium text-text-secondary hover:text-text-primary bg-surface-subtle hover:bg-brand-light px-3 py-2 rounded-lg border border-border transition-colors"
              title="Analyze another image"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>New Analysis</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
