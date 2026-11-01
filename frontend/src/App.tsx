import React, { useEffect, useMemo, useState } from 'react';
import { AlertCircle, Eye, EyeOff, Square, Tags, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
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
  summary: {
    whole_grain_count: number;
    whole_percent: number;
    whole_threshold_length: number;
    reference_length: number | null;
    touching_cluster_count: number;
    average_length: number | null;
    median_length: number | null;
    average_breadth: number | null;
    median_breadth: number | null;
    average_lb_ratio: number | null;
    median_lb_ratio: number | null;
  };
}

interface ProxyObject {
  id: string;
  category: 'rice' | 'foreign';
  bbox: [number, number, number, number];
  centroid: [number, number];
  pixel_count: number;
  mask_image: string;
  length?: number;
  effective_length?: number;
  ellipse_length?: number;
  breadth?: number;
  lb_ratio?: number;
  length_ratio?: number | null;
  mask_area?: number;
  solidity?: number;
  whole_label?: 'whole' | 'broken' | 'undetermined';
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
  const [viewerZoom, setViewerZoom] = useState(1);

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
      setSelectedObjectId(data.objects.find((object) => object.category === 'rice')?.id ?? data.objects[0]?.id ?? null);
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
            <div className="rounded-xl border border-border bg-surface-subtle px-4 py-3 text-xs leading-relaxed text-text-muted">
              Image-based estimates use grain counts and proxy-scale measurements. Whole-grain status uses the fresh proxy’s image-derived reference length and the 75% length cutoff.
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
              <MetricCard label="Total grains" value={result.rice_count} detail={`${result.summary.touching_cluster_count} clusters sent for splitting`} />
              <MetricCard label="Whole grains" value={result.summary.whole_grain_count} detail={`≥ 75% length${result.summary.reference_length ? ` · ref ${result.summary.reference_length.toFixed(1)} px` : ' · reference unavailable'}`} tone="green" />
              <MetricCard label="Foreign objects" value={result.foreign_matter_count} detail="Non-rice objects detected" tone="amber" />
              <MetricCard label="Whole %" value={`${result.summary.whole_percent.toFixed(2)}%`} detail="Whole kernels / rice grains" tone="green" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MetricCard label="Avg length" value={formatMetric(result.summary.average_length)} unit="px" detail={`Median: ${formatMetric(result.summary.median_length)}`} compact />
              <MetricCard label="Avg breadth" value={formatMetric(result.summary.average_breadth)} unit="px" detail={`Median: ${formatMetric(result.summary.median_breadth)}`} compact />
              <MetricCard label="Avg L/B ratio" value={formatMetric(result.summary.average_lb_ratio)} detail={`Median: ${formatMetric(result.summary.median_lb_ratio)}`} compact />
            </div>

            <div className="rounded-2xl border border-border bg-surface p-4 sm:p-6 shadow-sm">
              <div className="flex items-center justify-between gap-3 mb-3">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-brand">Per-grain table</p>
                  <h2 className="text-base font-semibold text-text-primary">Grain Population Overview</h2>
                </div>
                <span className="text-xs text-text-muted">{result.rice_count} detected grains</span>
              </div>
              <div className="max-h-[310px] overflow-auto rounded-xl border border-border">
                <table className="w-full min-w-[700px] text-left text-xs">
                  <thead className="sticky top-0 bg-surface-subtle text-[10px] uppercase tracking-wide text-text-muted">
                    <tr>
                      <th className="px-3 py-2.5">Grain #</th>
                      <th className="px-3 py-2.5">Status</th>
                      <th className="px-3 py-2.5">Effective length (pixels)</th>
                      <th className="px-3 py-2.5">Length ratio</th>
                      <th className="px-3 py-2.5">Breadth (pixels)</th>
                      <th className="px-3 py-2.5">L/B ratio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.objects.filter((object) => object.category === 'rice').map((grain) => {
                      const selected = grain.id === selectedObjectId;
                      const ratio = grain.length_ratio;
                      return (
                        <tr key={grain.id} onClick={() => setSelectedObjectId(grain.id)} className={`cursor-pointer border-t border-border transition hover:bg-brand-light/60 ${selected ? 'bg-brand-light/70' : 'bg-surface'}`}>
                          <td className="px-3 py-2.5 font-semibold">{grain.id}</td>
                          <td className="px-3 py-2.5"><StatusBadge status={grain.whole_label ?? 'undetermined'} /></td>
                          <td className="px-3 py-2.5 font-medium">{formatMetric(grain.effective_length ?? grain.length ?? null)}</td>
                          <td className="px-3 py-2.5">{ratio == null ? '—' : `${(ratio * 100).toFixed(1)}%`}</td>
                          <td className="px-3 py-2.5">{formatMetric(grain.breadth ?? null)}</td>
                          <td className="px-3 py-2.5">{formatMetric(grain.lb_ratio ?? null)}</td>
                        </tr>
                      );
                    })}
                    {!result.rice_count && <tr><td colSpan={6} className="px-3 py-8 text-center text-text-muted">No rice grains detected.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>

            {selectedObject && (
              <div className="rounded-2xl border border-border bg-surface p-4 sm:p-6 shadow-sm">
                <div className="flex items-center justify-between gap-3 border-b border-border pb-3 mb-4">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-widest text-brand">Per-grain inspector</p>
                    <h2 className="text-lg font-semibold text-text-primary">{selectedObject.category === 'rice' ? `Grain ${selectedObject.id}` : `Foreign object ${selectedObject.id}`}</h2>
                  </div>
                  {selectedObject.category === 'rice' && <StatusBadge status={selectedObject.whole_label ?? 'undetermined'} />}
                  {selectedObject.category === 'foreign' && <span className="rounded-full border border-red-200 bg-red-50 px-3 py-1 text-xs font-semibold text-red-700">FOREIGN MATTER</span>}
                </div>

                {selectedObject.category === 'rice' ? (
                  <>
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-3 mb-4">
                      <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-text-muted">Classification status</p>
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                        <InspectorStat label="Effective length" value={`${formatMetric(selectedObject.effective_length ?? selectedObject.length ?? null)} px`} />
                        <InspectorStat label="Reference length" value={`${formatMetric(result.summary.reference_length)} px`} />
                        <InspectorStat label="Length ratio" value={selectedObject.length_ratio == null ? '—' : `${(selectedObject.length_ratio * 100).toFixed(1)}%`} />
                        <InspectorStat label="Threshold" value="75.0% (3/4 rule)" />
                      </div>
                      <p className="mt-2 text-[10px] text-text-secondary">Basis: length ratio compared with the fresh proxy’s image-derived reference.</p>
                    </div>
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-text-secondary">Geometric measurements</p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                      <InspectorStat label="Ellipse length" value={`${formatMetric(selectedObject.ellipse_length ?? null)} px`} />
                      <InspectorStat label="Breadth" value={`${formatMetric(selectedObject.breadth ?? null)} px`} />
                      <InspectorStat label="L/B ratio" value={formatMetric(selectedObject.lb_ratio ?? null)} />
                      <InspectorStat label="Mask area" value={`${(selectedObject.mask_area ?? selectedObject.pixel_count).toLocaleString()} px²`} />
                      <InspectorStat label="Solidity" value={formatMetric(selectedObject.solidity ?? null)} />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                      <InspectorStat label="Bounding box" value={`[${selectedObject.bbox.map((value) => Math.round(value)).join(', ')}]`} mono />
                      <InspectorStat label="Centroid" value={`(${selectedObject.centroid.map((value) => value.toFixed(1)).join(', ')})`} mono />
                    </div>
                  </>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                    <InspectorStat label="Mask area" value={`${selectedObject.pixel_count.toLocaleString()} px²`} />
                    <InspectorStat label="Bounding box" value={`[${selectedObject.bbox.map((value) => Math.round(value)).join(', ')}]`} mono />
                    <InspectorStat label="Centroid" value={`(${selectedObject.centroid.map((value) => value.toFixed(1)).join(', ')})`} mono />
                  </div>
                )}
              </div>
            )}

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-5 items-start">
              <div className="rounded-2xl border border-border bg-surface p-4 sm:p-6 shadow-sm min-w-0">
                <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                  <div>
                    <h1 className="text-lg font-semibold text-text-primary">Annotated inspection</h1>
                    <div className="mt-1 flex flex-wrap gap-3 text-[10px] text-text-secondary">
                      <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-blue-600" />Whole</span>
                      <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-yellow-400" />Broken</span>
                      <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-red-600" />Foreign</span>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <select value={selectedObjectId ?? ''} onChange={(event) => setSelectedObjectId(event.target.value || null)} className="max-w-40 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-xs text-text-primary">
                      {result.objects.map((object) => <option key={object.id} value={object.id}>{object.category === 'rice' ? 'Grain' : 'Foreign'} {object.id}</option>)}
                    </select>
                    <ToggleButton active={showIds} onClick={() => setShowIds(!showIds)} icon={<Tags className="w-4 h-4" />} label={`IDs ${showIds ? 'ON' : 'OFF'}`} />
                    <ToggleButton active={showMasks} onClick={() => setShowMasks(!showMasks)} icon={showMasks ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />} label={`Masks ${showMasks ? 'ON' : 'OFF'}`} />
                    <ToggleButton active={showBoxes} onClick={() => setShowBoxes(!showBoxes)} icon={<Square className="w-4 h-4" />} label={`Boxes ${showBoxes ? 'ON' : 'OFF'}`} />
                    <div className="flex items-center rounded-lg border border-border bg-surface">
                      <button type="button" onClick={() => setViewerZoom((zoom) => Math.max(0.5, +(zoom - 0.25).toFixed(2)))} className="p-1.5 text-text-secondary hover:text-text-primary" aria-label="Zoom out"><ZoomOut className="w-4 h-4" /></button>
                      <button type="button" onClick={() => setViewerZoom(1)} className="min-w-12 px-1 text-[10px] text-text-secondary">{Math.round(viewerZoom * 100)}%</button>
                      <button type="button" onClick={() => setViewerZoom((zoom) => Math.min(3, +(zoom + 0.25).toFixed(2)))} className="p-1.5 text-text-secondary hover:text-text-primary" aria-label="Zoom in"><ZoomIn className="w-4 h-4" /></button>
                      <button type="button" onClick={() => setViewerZoom(1)} className="p-1.5 text-text-secondary hover:text-text-primary" aria-label="Reset zoom"><Maximize2 className="w-4 h-4" /></button>
                    </div>
                  </div>
                </div>

                <div className="bg-[#171a19] rounded-xl p-2 sm:p-4 overflow-auto text-center max-h-[78vh]">
                  <div className="relative inline-block align-middle max-w-full" style={{ transform: `scale(${viewerZoom})`, transformOrigin: 'top center' }}>
                    <img
                      src={showMasks ? result.masked_image : result.original_image}
                      alt={showMasks ? 'Image with individual object masks' : 'Original uploaded image'}
                      className="block max-h-[72vh] max-w-full w-auto rounded-lg"
                    />
                    {result.objects.map((object) => {
                      const [x, y, width, height] = object.bbox;
                      const [centerX, centerY] = object.centroid;
                      const isSelected = object.id === selectedObjectId;
                      const color = objectColor(object);
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
                  <ObjectList title="Rice grains" objects={result.objects.filter((object) => object.category === 'rice')} selectedId={selectedObjectId} onSelect={setSelectedObjectId} />
                  <ObjectList title="Foreign matter" objects={result.objects.filter((object) => object.category === 'foreign')} selectedId={selectedObjectId} onSelect={setSelectedObjectId} />
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

function formatMetric(value: number | null): string {
  return value === null ? '—' : value.toFixed(2);
}

function MetricCard({ label, value, detail, tone = 'neutral', unit, compact = false }: {
  label: string;
  value: string | number;
  detail: string;
  tone?: 'neutral' | 'green' | 'amber';
  unit?: string;
  compact?: boolean;
}) {
  const colors = {
    neutral: 'border-border bg-surface text-text-primary',
    green: 'border-emerald-200 bg-white text-emerald-600',
    amber: 'border-amber-200 bg-amber-50/70 text-amber-700',
  }[tone];
  return (
    <div className={`rounded-2xl border p-4 sm:p-5 shadow-sm ${colors}`}>
      <p className={`font-semibold uppercase tracking-wide text-text-muted ${compact ? 'text-[11px]' : 'text-xs'}`}>{label}</p>
      <div className="mt-1 flex items-baseline gap-2">
        <p className={`font-bold leading-none ${compact ? 'text-2xl' : 'text-4xl'}`}>{value}</p>
        {unit && <span className="text-xs text-text-muted">{unit}</span>}
      </div>
      <p className={`mt-2 text-xs ${tone === 'neutral' ? 'text-text-muted' : 'opacity-90'}`}>{detail}</p>
    </div>
  );
}

function StatusBadge({ status }: { status: 'whole' | 'broken' | 'undetermined' }) {
  const style = status === 'whole'
    ? 'border-blue-200 bg-blue-50 text-blue-700'
    : status === 'broken'
      ? 'border-yellow-300 bg-yellow-50 text-yellow-800'
      : 'border-border bg-surface-subtle text-text-secondary';
  return <span className={`inline-flex rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${style}`}>{status}</span>;
}

function objectColor(object: ProxyObject): string {
  if (object.category === 'foreign') return '#dc2626';
  if (object.whole_label === 'whole') return '#2563eb';
  if (object.whole_label === 'broken') return '#eab308';
  return '#6b7280';
}

function InspectorStat({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-surface-subtle px-3 py-2">
      <p className="text-[9px] font-semibold uppercase tracking-wide text-text-muted">{label}</p>
      <p className={`mt-0.5 text-xs font-semibold text-text-primary ${mono ? 'font-mono font-normal' : ''}`}>{value}</p>
    </div>
  );
}

function ToggleButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${active ? 'border-brand bg-brand-light text-brand' : 'border-border bg-surface text-text-secondary'}`}>
      {icon}<span>{label}</span>
    </button>
  );
}

function ObjectList({ title, objects, selectedId, onSelect }: { title: string; objects: ProxyObject[]; selectedId: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between text-xs font-medium text-text-secondary mb-2">
        <span>{title}</span><span>{objects.length}</span>
      </div>
      {objects.length ? (
        <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pr-1">
          {objects.map((object) => (
            <button key={object.id} type="button" onClick={() => onSelect(object.id)} className="rounded-md border px-2 py-1 text-xs font-semibold" style={{ backgroundColor: object.id === selectedId ? objectColor(object) : `${objectColor(object)}18`, color: object.id === selectedId ? '#fff' : objectColor(object), borderColor: `${objectColor(object)}66` }}>
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
