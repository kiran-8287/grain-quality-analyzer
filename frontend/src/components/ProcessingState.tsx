import React from 'react';
import { Loader2 } from 'lucide-react';

export const ProcessingState: React.FC = () => {
  return (
    <div className="max-w-xl mx-auto px-4 py-16 text-center">
      <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-brand-light border border-border flex items-center justify-center text-brand">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>

      <h2 className="text-xl font-bold text-text-primary mb-2">Creating masked image</h2>
      <p className="text-xs text-text-secondary mb-8 max-w-sm mx-auto">
        Running the fresh auto-label proxy on your upload...
      </p>
    </div>
  );
};
