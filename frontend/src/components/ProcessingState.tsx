import React from 'react';
import { Loader2 } from 'lucide-react';

interface ProcessingStateProps {
  elapsedSeconds: number;
  filename?: string;
}

export const ProcessingState: React.FC<ProcessingStateProps> = ({ elapsedSeconds, filename }) => {
  return (
    <div aria-busy="true" className="max-w-xl mx-auto px-4 py-16 text-center">
      <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-brand-light border border-border flex items-center justify-center text-brand">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>

      <h2 role="status" aria-live="polite" className="text-xl font-bold text-text-primary mb-2">Analyzing rice sample</h2>
      <p className="text-sm text-text-secondary mb-2 max-w-sm mx-auto">
        Separating grain masks and counting detected objects. Detailed images can take a little longer.
      </p>
      {filename && <p className="mx-auto max-w-sm truncate text-xs text-text-muted" title={filename}>{filename}</p>}
      <p aria-hidden="true" className="mt-4 font-mono text-sm tabular-nums text-brand">Elapsed {formatElapsed(elapsedSeconds)}</p>
      <p className="mt-3 text-xs text-text-muted">Keep this page open. Your selected image will stay available if you need to retry.</p>
    </div>
  );
};

function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return minutes ? `${minutes}m ${remainder.toString().padStart(2, '0')}s` : `${remainder}s`;
}
