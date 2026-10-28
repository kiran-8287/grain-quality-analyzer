import React, { useEffect, useMemo, useState } from 'react';
import { AlertCircle, Eye, EyeOff, Square, Tags } from 'lucide-react';
import { Navbar } from './components/Navbar';
import { UploadSection } from './components/UploadSection';
import { ProcessingState } from './components/ProcessingState';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

interface ProxyResult {
  masked_image: string;
  original_image: string;
  image_width: number;
  image_height: number;
  rice_count: number;
  foreign_matter_count: number;
  objects: ProxyObject[];
}

interface ProxyObject {
  id: string;
  category: 'rice' | 'foreign';
  bbox: [number, number, number, number];
  centroid: [number, number];
  pixel_count: number;
  mask_image: string;
}

export const App: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ProxyResult | null>(null);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [showMasks, setShowMasks] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showIds, setShowIds] = useState(true);

  const selectedObject = useMemo(
    () => result?.objects.find((object) => object.id === selectedObjectId) ?? null,
    [result, selectedObjectId],
  );

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        setBackendHealthy(response.ok && (await response.json()).status === 'healthy');
      } catch {
        setBackendHealthy(false);
      }
    };

    void checkHealth();
    const interval = window.setInterval(checkHealth, 10000);
    return () => window.clearInterval(interval);
  }, []);

  const handleAnalyze = async (file: File) => {
    setIsLoading(true);
    setErrorMessage(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/analyze-proxy`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }
      const data = (await response.json()) as ProxyResult;
      setResult(data);
      setSelectedObjectId(data.objects[0]?.id ?? null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unable to analyze this image.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setErrorMessage(null);
    setSelectedObjectId(null);
  };

  return (
    <div className="min-h-screen bg-surface-page text-text-primary flex flex-col">
      <Navbar onReset={handleReset} hasResult={result !== null} isBackendHealthy={backendHealthy} />

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        {errorMessage && (
          <div className="mb-6 bg-error-bg border border-error/30 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-error shrink-0 mt-0.5" />
            <p className="text-sm text-text-secondary">{errorMessage}</p>
          </div>
        )}

        {isLoading && <ProcessingState />}

        {!isLoading && !result && <UploadSection onAnalyze={handleAnalyze} isLoading={isLoading} />}

        {!isLoading && result && (
          <section className="animate-fadeIn space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="rounded-2xl border border-border bg-surface p-5 text-center shadow-sm flex items-center justify-between">
                <p className="text-sm text-text-secondary">Rice grains</p>
                <p className="text-4xl font-bold text-text-primary">{result.rice_count}</p>
              </div>
              <div className="rounded-2xl border border-border bg-surface p-5 text-center shadow-sm flex items-center justify-between">
                <p className="text-sm text-text-secondary">Foreign matter</p>
                <p className="text-4xl font-bold text-text-primary">{result.foreign_matter_count}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-5 items-start">
              <div className="rounded-2xl border border-border bg-surface p-4 sm:p-6 shadow-sm min-w-0">
                <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                  <div>
                    <h1 className="text-lg font-semibold text-text-primary">Masked output image</h1>
                    <p className="text-xs text-text-muted mt-1">Rice masks are blue; foreign matter masks are red.</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <ToggleButton active={showMasks} onClick={() => setShowMasks(!showMasks)} icon={showMasks ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />} label="Masks" />
                    <ToggleButton active={showBoxes} onClick={() => setShowBoxes(!showBoxes)} icon={<Square className="w-4 h-4" />} label="Boxes" />
                    <ToggleButton active={showIds} onClick={() => setShowIds(!showIds)} icon={<Tags className="w-4 h-4" />} label="IDs" />
                  </div>
                </div>

                <div className="bg-surface-subtle rounded-xl p-2 sm:p-4 overflow-auto text-center">
                  <div className="relative inline-block align-middle max-w-full">
                    <img
                      src={showMasks ? result.masked_image : result.original_image}
                      alt={showMasks ? 'Image with individual object masks' : 'Original uploaded image'}
                      className="block max-h-[72vh] max-w-full w-auto rounded-lg"
                    />
                    {result.objects.map((object) => {
                      const [x, y, width, height] = object.bbox;
                      const [centerX, centerY] = object.centroid;
                      const isSelected = object.id === selectedObjectId;
                      const color = object.category === 'rice' ? '#2563eb' : '#dc2626';
                      return (
                        <React.Fragment key={object.id}>
                          {showBoxes && (
                            <button
                              type="button"
                              aria-label={`Select ${object.category} ${object.id}`}
                              onClick={() => setSelectedObjectId(object.id)}
                              className="absolute border-2 bg-transparent"
                              style={{ left: `${(x / result.image_width) * 100}%`, top: `${(y / result.image_height) * 100}%`, width: `${(width / result.image_width) * 100}%`, height: `${(height / result.image_height) * 100}%`, borderColor: color, boxShadow: isSelected ? `0 0 0 2px white, 0 0 0 4px ${color}` : undefined }}
                            />
                          )}
                          {showIds && (
                            <button
                              type="button"
                              onClick={() => setSelectedObjectId(object.id)}
                              className="absolute -translate-x-1/2 -translate-y-1/2 rounded-md px-1.5 py-0.5 text-[10px] font-bold text-white shadow-md ring-1 ring-white/80"
                              style={{ left: `${(centerX / result.image_width) * 100}%`, top: `${(centerY / result.image_height) * 100}%`, backgroundColor: color, zIndex: isSelected ? 20 : 10 }}
                            >
                              {object.id}
                            </button>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </div>
                </div>
              </div>

              <aside className="space-y-4">
                <div className="rounded-2xl border border-border bg-surface p-4 shadow-sm">
                  <h2 className="font-semibold text-text-primary mb-3">Detected objects</h2>
                  <ObjectList title="Rice grains" objects={result.objects.filter((object) => object.category === 'rice')} selectedId={selectedObjectId} onSelect={setSelectedObjectId} color="blue" />
                  <ObjectList title="Foreign matter" objects={result.objects.filter((object) => object.category === 'foreign')} selectedId={selectedObjectId} onSelect={setSelectedObjectId} color="red" />
                </div>

                {selectedObject && (
                  <div className="rounded-2xl border border-border bg-surface p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <h2 className="font-semibold text-text-primary">{selectedObject.id}</h2>
                      <span className={`text-xs font-medium capitalize ${selectedObject.category === 'rice' ? 'text-blue-600' : 'text-red-600'}`}>
                        {selectedObject.category === 'foreign' ? 'foreign matter' : 'rice grain'}
                      </span>
                    </div>
                    <div className="rounded-xl bg-surface-subtle p-4 flex items-center justify-center min-h-36">
                      <img src={selectedObject.mask_image} alt={`Isolated mask for ${selectedObject.id}`} className="max-h-40 max-w-full object-contain" />
                    </div>
                    <p className="mt-2 text-xs text-text-muted">Isolated mask · {selectedObject.pixel_count.toLocaleString()} pixels</p>
                  </div>
                )}
              </aside>
            </div>

            <div className="text-center">
              <button
                type="button"
                onClick={handleReset}
                className="rounded-xl bg-brand px-6 py-3 text-sm font-semibold text-white transition hover:bg-brand-dark"
              >
                Analyze another image
              </button>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};

function ToggleButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${active ? 'border-brand bg-brand-light text-brand' : 'border-border bg-surface text-text-secondary'}`}>
      {icon}<span>{label}</span>
    </button>
  );
}

function ObjectList({ title, objects, selectedId, onSelect, color }: { title: string; objects: ProxyObject[]; selectedId: string | null; onSelect: (id: string) => void; color: 'blue' | 'red' }) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between text-xs font-medium text-text-secondary mb-2">
        <span>{title}</span><span>{objects.length}</span>
      </div>
      {objects.length ? (
        <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pr-1">
          {objects.map((object) => (
            <button key={object.id} type="button" onClick={() => onSelect(object.id)} className={`rounded-md border px-2 py-1 text-xs font-semibold ${object.id === selectedId ? (color === 'blue' ? 'border-blue-600 bg-blue-600 text-white' : 'border-red-600 bg-red-600 text-white') : (color === 'blue' ? 'border-blue-200 bg-blue-50 text-blue-700' : 'border-red-200 bg-red-50 text-red-700')}`}>
              {object.id}
            </button>
          ))}
        </div>
      ) : (
        <p className="text-xs text-text-muted">No objects detected.</p>
      )}
    </div>
  );
}
