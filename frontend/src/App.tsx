import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Download, Eye, EyeOff, FileSpreadsheet, Square, Tags, ZoomIn, ZoomOut, Maximize2, Move } from 'lucide-react';
import { Navbar } from './components/Navbar';
import { UploadSection } from './components/UploadSection';
import { ProcessingState } from './components/ProcessingState';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

interface RiceAnalysisResult {
  masked_image: string;
  original_image: string;
  image_width: number;
  image_height: number;
  rice_count: number;
  foreign_matter_count: number;
  objects: DetectedObject[];
  summary: {
    whole_grain_count: number;
    broken_grain_count: number;
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

interface DetectedObject {
  id: string;
  category: 'rice' | 'foreign';
  bbox: [number, number, number, number];
  centroid: [number, number];
  mask_polygon: [number, number][];
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
  const [result, setResult] = useState<RiceAnalysisResult | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analyzedFile, setAnalyzedFile] = useState<{ name: string; size: number } | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [grainSearch, setGrainSearch] = useState('');
  const [grainStatusFilter, setGrainStatusFilter] = useState<'all' | 'whole' | 'broken' | 'undetermined'>('all');
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [showMasks, setShowMasks] = useState(true);
  const [showBoxes, setShowBoxes] = useState(false);
  const [showIds, setShowIds] = useState(false);
  const [viewerZoom, setViewerZoom] = useState(1);
  const [viewerOrigin, setViewerOrigin] = useState({ x: 50, y: 50 });
  const [panEnabled, setPanEnabled] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const inspectorRef = useRef<HTMLDivElement>(null);
  const imageViewportRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ pointerId: number; startX: number; startY: number; scrollLeft: number; scrollTop: number; moved: boolean } | null>(null);
  const didPanRef = useRef(false);

  const selectedObject = useMemo(
    () => result?.objects.find((object) => object.id === selectedObjectId) ?? null,
    [result, selectedObjectId],
  );
  const visibleGrains = useMemo(() => {
    const query = grainSearch.trim().toLowerCase();
    return (result?.objects ?? []).filter((object) => {
      if (object.category !== 'rice') return false;
      if (grainStatusFilter !== 'all' && object.whole_label !== grainStatusFilter) return false;
      return !query || object.id.toLowerCase().includes(query);
    });
  }, [result, grainSearch, grainStatusFilter]);

  useEffect(() => {
    if (!isLoading) {
      setElapsedSeconds(0);
      return;
    }
    const startedAt = Date.now();
    const timer = window.setInterval(() => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [isLoading]);

  const selectFromImage = (id: string) => {
    setSelectedObjectId(id);
    window.requestAnimationFrame(() => {
      inspectorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  };

  const handleImageClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (didPanRef.current) {
      didPanRef.current = false;
      return;
    }
    if (!result) return;
    const image = event.currentTarget.querySelector('img');
    if (!image) return;
    const rect = image.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const px = ((event.clientX - rect.left) / rect.width) * result.image_width;
    const py = ((event.clientY - rect.top) / rect.height) * result.image_height;
    const hits = result.objects.filter((object) =>
      object.mask_polygon.length >= 3 && pointInPolygon(px, py, object.mask_polygon),
    );
    if (!hits.length) return;

    hits.sort((a, b) => {
      const distanceA = Math.hypot(a.centroid[0] - px, a.centroid[1] - py);
      const distanceB = Math.hypot(b.centroid[0] - px, b.centroid[1] - py);
      return distanceA - distanceB;
    });
    selectFromImage(hits[0].id);
  };

  const handlePanStart = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!panEnabled) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    const viewport = imageViewportRef.current;
    if (!viewport) return;
    panRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop,
      moved: false,
    };
    viewport.setPointerCapture(event.pointerId);
  };

  const handlePanMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    const viewport = imageViewportRef.current;
    if (!pan || !viewport || pan.pointerId !== event.pointerId) return;
    const dx = event.clientX - pan.startX;
    const dy = event.clientY - pan.startY;
    if (Math.abs(dx) + Math.abs(dy) > 4) {
      pan.moved = true;
      setIsPanning(true);
    }
    if (pan.moved) {
      event.preventDefault();
      viewport.scrollLeft = pan.scrollLeft - dx;
      viewport.scrollTop = pan.scrollTop - dy;
    }
  };

  const handlePanEnd = (event: React.PointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    if (!pan || pan.pointerId !== event.pointerId) return;
    didPanRef.current = pan.moved;
    panRef.current = null;
    setIsPanning(false);
    if (imageViewportRef.current?.hasPointerCapture(event.pointerId)) {
      imageViewportRef.current.releasePointerCapture(event.pointerId);
    }
  };

  const setCenteredZoom = (nextZoom: number) => {
    setViewerOrigin({ x: 50, y: 50 });
    setViewerZoom(nextZoom);
    if (nextZoom === 1) {
      window.requestAnimationFrame(() => {
        if (imageViewportRef.current) {
          imageViewportRef.current.scrollLeft = 0;
          imageViewportRef.current.scrollTop = 0;
        }
      });
    }
  };

  useEffect(() => {
    const viewport = imageViewportRef.current;
    if (!viewport || !result) return;

    const onWheel = (event: WheelEvent) => {
      const image = viewport.querySelector('img');
      if (!image) return;
      const imageRect = image.getBoundingClientRect();
      if (event.clientX < imageRect.left || event.clientX > imageRect.right || event.clientY < imageRect.top || event.clientY > imageRect.bottom) return;

      // A native non-passive listener is required to stop wheel events from
      // scrolling the document while the cursor is over the image.
      event.preventDefault();
      const anchorX = Math.min(1, Math.max(0, (event.clientX - imageRect.left) / imageRect.width));
      const anchorY = Math.min(1, Math.max(0, (event.clientY - imageRect.top) / imageRect.height));
      const nextZoom = Math.min(3, Math.max(0.5, viewerZoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15)));
      if (nextZoom === viewerZoom) return;

      setViewerOrigin({ x: anchorX * 100, y: anchorY * 100 });
      setViewerZoom(nextZoom);
      window.requestAnimationFrame(() => {
        const nextRect = image.getBoundingClientRect();
        viewport.scrollLeft += nextRect.left + anchorX * nextRect.width - event.clientX;
        viewport.scrollTop += nextRect.top + anchorY * nextRect.height - event.clientY;
      });
    };

    viewport.addEventListener('wheel', onWheel, { passive: false });
    return () => viewport.removeEventListener('wheel', onWheel);
  }, [result, viewerZoom]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        setBackendStatus(response.ok && (await response.json()).status === 'healthy' ? 'online' : 'offline');
      } catch {
        setBackendStatus('offline');
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
    setAnalyzedFile({ name: file.name, size: file.size });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/rice-quality/analyze`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Analysis failed' }));
        const details = errorData.detail || `Server error (${response.status})`;
        throw new Error(response.status === 413
          ? 'This image exceeds the upload size or pixel-dimension limit. Choose a smaller image and try again.'
          : details);
      }
      const data = (await response.json()) as RiceAnalysisResult;
      setResult(data);
      setSelectedObjectId(data.objects.find((object) => object.category === 'rice')?.id ?? data.objects[0]?.id ?? null);
    } catch (error) {
      const message = error instanceof Error ? error.message : '';
      setErrorMessage(message === 'Failed to fetch'
        ? 'Could not connect to the analysis service. Check your connection and try again.'
        : message || 'Unable to analyze this image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setErrorMessage(null);
    setSelectedObjectId(null);
    setAnalyzedFile(null);
    setGrainSearch('');
    setGrainStatusFilter('all');
    setViewerZoom(1);
    setViewerOrigin({ x: 50, y: 50 });
  };

  const downloadDataUrl = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const downloadCsv = () => {
    if (!result) return;
    const header = ['ID', 'Type', 'Status', 'Length (px)', 'Length ratio (%)', 'Breadth (px)', 'L/B ratio', 'Mask area (px2)', 'X', 'Y', 'Width', 'Height'];
    const escapeCsv = (value: string | number | null | undefined) => `"${String(value ?? '').replace(/"/g, '""')}"`;
    const rows = result.objects.map((object) => [
      object.id,
      object.category === 'rice' ? 'rice' : 'foreign matter',
      object.category === 'rice' ? object.whole_label : '',
      object.length ?? '',
      object.length_ratio == null ? '' : (object.length_ratio * 100).toFixed(1),
      object.breadth ?? '',
      object.lb_ratio ?? '',
      object.pixel_count,
      ...object.bbox,
    ].map(escapeCsv).join(','));
    const csv = [header.map(escapeCsv).join(','), ...rows].join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `rice-analysis-${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="min-h-screen bg-surface-page text-text-primary flex flex-col">
      <a href="#main-content" className="skip-link">Skip to main content</a>
      <Navbar onReset={handleReset} hasResult={result !== null} backendStatus={backendStatus} />

      <main id="main-content" className="flex-1 min-w-0 max-w-6xl mx-auto w-full px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6 lg:py-8">
        {errorMessage && (
          <div role="alert" className="mb-6 bg-error-bg border border-error/30 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-error shrink-0 mt-0.5" />
            <p className="text-sm text-text-secondary">{errorMessage}</p>
          </div>
        )}

        {isLoading && <ProcessingState elapsedSeconds={elapsedSeconds} filename={analyzedFile?.name} />}

        {!result && <div hidden={isLoading}><UploadSection onAnalyze={handleAnalyze} isLoading={isLoading} /></div>}

        {!isLoading && result && (
          <section className="animate-fadeIn min-w-0 space-y-4 sm:space-y-6">
            <div className="rounded-xl border border-border bg-surface-subtle px-4 py-3 text-xs leading-relaxed text-text-muted">
              Counts and measurements are image-based estimates. Whole/broken labels compare each grain with a reference length estimated from this image; they are not a certified weight-based grade.
            </div>

            <div className="flex flex-col gap-3 rounded-2xl border border-border bg-surface p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <p className="text-sm font-semibold text-text-primary">Analysis results</p>
                <p className="mt-0.5 truncate text-xs text-text-secondary" title={analyzedFile?.name}>{analyzedFile?.name ?? 'Uploaded image'}{analyzedFile ? ` · ${(analyzedFile.size / 1024).toFixed(0)} KB` : ''}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <button type="button" onClick={() => downloadDataUrl(result.masked_image, 'rice-analysis-masked.jpg')} className="inline-flex min-h-10 items-center justify-center gap-2 rounded-lg bg-brand px-3 py-2 text-xs font-semibold text-white hover:bg-brand-dark">
                  <Download className="h-4 w-4" /> Masked image
                </button>
                <button type="button" onClick={downloadCsv} className="inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-xs font-semibold text-text-secondary hover:bg-surface-subtle">
                  <FileSpreadsheet className="h-4 w-4" /> Grain data CSV
                </button>
                <button type="button" onClick={handleReset} className="min-h-10 rounded-lg border border-border bg-surface px-3 py-2 text-xs font-semibold text-text-secondary hover:bg-surface-subtle">New analysis</button>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-2 sm:gap-4">
              <MetricCard label="Total grains" value={result.rice_count} detail={`${result.summary.touching_cluster_count} clusters sent for splitting`} />
              <MetricCard label="Whole grains" value={result.summary.whole_grain_count} detail={`≥ 75% length${result.summary.reference_length ? ` · ref ${result.summary.reference_length.toFixed(1)} px` : ' · reference unavailable'}`} tone="green" />
              <MetricCard label="Broken grains" value={result.summary.broken_grain_count} detail={result.summary.reference_length ? '< 75% length' : 'Reference unavailable'} tone="yellow" />
              <MetricCard label="Foreign objects" value={result.foreign_matter_count} detail="Non-rice objects detected" tone="red" />
              <MetricCard label="Whole %" value={`${result.summary.whole_percent.toFixed(2)}%`} detail="Whole kernels / rice grains" tone="green" />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 sm:gap-4">
              <MetricCard label="Avg length" value={formatMetric(result.summary.average_length)} unit="px" detail={`Median: ${formatMetric(result.summary.median_length)}`} compact />
              <MetricCard label="Avg breadth" value={formatMetric(result.summary.average_breadth)} unit="px" detail={`Median: ${formatMetric(result.summary.median_breadth)}`} compact />
              <MetricCard label="Avg L/B ratio" value={formatMetric(result.summary.average_lb_ratio)} detail={`Median: ${formatMetric(result.summary.median_lb_ratio)}`} compact />
            </div>

            <details className="group rounded-xl border border-border bg-surface px-4 py-3 text-xs text-text-secondary">
              <summary className="cursor-pointer font-semibold text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2">How are whole and broken grains classified?</summary>
              <p className="mt-2 max-w-3xl leading-relaxed">The analysis estimates a reference length from grains in this image. A detected rice object at least 75% of that reference is labeled whole; a shorter object is labeled broken. Measurements are in pixels and depend on image scale. This count-based estimate does not replace a laboratory or weight-based regulatory test.</p>
            </details>

            <div className="rounded-2xl border border-border bg-surface p-4 sm:p-6 shadow-sm">
              <div className="flex flex-col gap-3 mb-3 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-brand">Per-grain table</p>
                  <h2 className="text-base font-semibold text-text-primary">Grain Population Overview</h2>
                  <span className="text-xs text-text-muted">Showing {visibleGrains.length} of {result.rice_count} grains</span>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(150px,1fr)_auto]">
                  <label className="sr-only" htmlFor="grain-search">Find a grain by ID</label>
                  <input id="grain-search" type="search" value={grainSearch} onChange={(event) => setGrainSearch(event.target.value)} placeholder="Find grain ID" className="min-h-10 rounded-lg border border-border bg-surface px-3 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand" />
                  <label className="sr-only" htmlFor="grain-status-filter">Filter grains by status</label>
                  <select id="grain-status-filter" value={grainStatusFilter} onChange={(event) => setGrainStatusFilter(event.target.value as typeof grainStatusFilter)} className="min-h-10 rounded-lg border border-border bg-surface px-3 text-sm text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand">
                    <option value="all">All statuses</option>
                    <option value="whole">Whole</option>
                    <option value="broken">Broken</option>
                    <option value="undetermined">Undetermined</option>
                  </select>
                </div>
              </div>
              <div role="region" aria-label="Grain population table" tabIndex={0} className="max-h-[310px] overflow-auto overscroll-contain rounded-xl border border-border">
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
                    {visibleGrains.map((grain) => {
                      const selected = grain.id === selectedObjectId;
                      const ratio = grain.length_ratio;
                      const selectGrain = () => setSelectedObjectId(grain.id);
                      return (
                        <tr key={grain.id} tabIndex={0} aria-selected={selected} aria-label={`Select grain ${grain.id}, ${grain.whole_label ?? 'undetermined'}`} onClick={selectGrain} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectGrain(); } }} className={`cursor-pointer border-t border-border transition hover:bg-brand-light/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand ${selected ? 'bg-brand-light/70' : 'bg-surface'}`}>
                          <td className="px-3 py-2.5 font-semibold">{grain.id}</td>
                          <td className="px-3 py-2.5"><StatusBadge status={grain.whole_label ?? 'undetermined'} /></td>
                          <td className="px-3 py-2.5 font-medium">{formatMetric(grain.effective_length ?? grain.length ?? null)}</td>
                          <td className="px-3 py-2.5">{ratio == null ? '—' : `${(ratio * 100).toFixed(1)}%`}</td>
                          <td className="px-3 py-2.5">{formatMetric(grain.breadth ?? null)}</td>
                          <td className="px-3 py-2.5">{formatMetric(grain.lb_ratio ?? null)}</td>
                        </tr>
                      );
                    })}
                    {visibleGrains.length === 0 && <tr><td colSpan={6} className="px-3 py-8 text-center text-text-muted">{result.rice_count ? 'No grains match these filters.' : 'No rice grains detected.'}</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>

            {selectedObject && (
              <div ref={inspectorRef} className="min-w-0 rounded-2xl border border-border bg-surface p-3 sm:p-5 md:p-6 shadow-sm scroll-mt-24">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-border pb-3 mb-4">
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
                      <p className="mt-2 text-[10px] text-text-secondary">Basis: length ratio compared with the whole-grain reference estimated from this image.</p>
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

            <div className="grid min-w-0 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px] 2xl:grid-cols-[minmax(0,1fr)_320px] gap-3 sm:gap-5 items-start">
              <div className="min-w-0 rounded-2xl border border-border bg-surface p-3 sm:p-5 md:p-6 shadow-sm">
                <div className="mb-3 sm:mb-4 flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
                  <div className="min-w-0">
                    <h1 className="text-lg font-semibold text-text-primary">Annotated inspection</h1>
                    <div className="mt-1 flex flex-wrap gap-3 text-[10px] text-text-secondary">
                      <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-blue-600" />Whole</span>
                      <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-yellow-400" />Broken</span>
                      <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-red-600" />Foreign</span>
                    </div>
                    <p className="mt-1 text-[10px] text-text-muted">Scroll over the image to zoom at the pointer. Turn Pan on to drag the image.</p>
                  </div>
                  <div className="grid w-full grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:items-center xl:w-auto xl:justify-end">
                    <label className="sr-only" htmlFor="object-select">Select a detected object</label>
                    <select id="object-select" value={selectedObjectId ?? ''} onChange={(event) => setSelectedObjectId(event.target.value || null)} className="col-span-2 min-h-10 min-w-0 w-full rounded-lg border border-border bg-surface px-2.5 py-2 text-xs text-text-primary sm:col-span-1 sm:w-36">
                      {result.objects.map((object) => <option key={object.id} value={object.id}>{object.category === 'rice' ? 'Grain' : 'Foreign'} {object.id}</option>)}
                    </select>
                    <ToggleButton active={showIds} onClick={() => setShowIds(!showIds)} icon={<Tags className="w-4 h-4" />} label={`IDs ${showIds ? 'ON' : 'OFF'}`} />
                    <ToggleButton active={showMasks} onClick={() => setShowMasks(!showMasks)} icon={showMasks ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />} label={`Masks ${showMasks ? 'ON' : 'OFF'}`} />
                    <ToggleButton active={showBoxes} onClick={() => setShowBoxes(!showBoxes)} icon={<Square className="w-4 h-4" />} label={`Boxes ${showBoxes ? 'ON' : 'OFF'}`} />
                    <ToggleButton active={panEnabled} onClick={() => { setPanEnabled(!panEnabled); setIsPanning(false); }} icon={<Move className="w-4 h-4" />} label={`Pan ${panEnabled ? 'ON' : 'OFF'}`} />
                    <div className="col-span-2 flex w-full items-center justify-center rounded-lg border border-border bg-surface sm:col-span-1 sm:w-auto">
                      <button type="button" onClick={() => setCenteredZoom(Math.max(0.5, +(viewerZoom - 0.25).toFixed(2)))} className="p-1.5 text-text-secondary hover:text-text-primary" aria-label="Zoom out"><ZoomOut className="w-4 h-4" /></button>
                      <button type="button" onClick={() => setCenteredZoom(1)} className="min-w-12 px-1 text-[10px] text-text-secondary">{Math.round(viewerZoom * 100)}%</button>
                      <button type="button" onClick={() => setCenteredZoom(Math.min(3, +(viewerZoom + 0.25).toFixed(2)))} className="p-1.5 text-text-secondary hover:text-text-primary" aria-label="Zoom in"><ZoomIn className="w-4 h-4" /></button>
                      <button type="button" onClick={() => setCenteredZoom(1)} className="p-1.5 text-text-secondary hover:text-text-primary" aria-label="Reset zoom"><Maximize2 className="w-4 h-4" /></button>
                    </div>
                  </div>
                </div>

                <div
                  ref={imageViewportRef}
                  onClick={handleImageClick}
                  onPointerDown={handlePanStart}
                  onPointerMove={handlePanMove}
                  onPointerUp={handlePanEnd}
                  onPointerCancel={handlePanEnd}
                  onDragStart={(event) => event.preventDefault()}
                  className={`bg-[#171a19] rounded-xl p-1.5 sm:p-3 md:p-4 overflow-auto overscroll-contain text-center max-h-[70vh] sm:max-h-[78vh] ${panEnabled ? 'touch-none' : 'touch-pan-y'} select-none ${isPanning ? 'cursor-grabbing' : panEnabled ? 'cursor-grab' : 'cursor-crosshair'}`}
                >
                  <div className="relative inline-block align-middle max-w-full" style={{ transform: `scale(${viewerZoom})`, transformOrigin: `${viewerOrigin.x}% ${viewerOrigin.y}%` }}>
                    <img
                      src={showMasks ? result.masked_image : result.original_image}
                      alt={showMasks ? 'Image with individual object masks' : 'Original uploaded image'}
                      className="block max-h-[62vh] sm:max-h-[72vh] max-w-full w-auto rounded-lg"
                    />
                    {selectedObject && selectedObject.mask_polygon.length >= 3 && (
                      <svg
                        aria-hidden="true"
                        className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
                        viewBox={`0 0 ${result.image_width} ${result.image_height}`}
                        preserveAspectRatio="none"
                      >
                        <polygon
                          points={selectedObject.mask_polygon.map(([x, y]) => `${x},${y}`).join(' ')}
                          fill="none"
                          stroke="#ffffff"
                          strokeWidth="1.25"
                          strokeLinejoin="round"
                          vectorEffect="non-scaling-stroke"
                        />
                      </svg>
                    )}
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
                              onClick={(event) => { event.preventDefault(); event.stopPropagation(); selectFromImage(object.id); }}
                              className="absolute border-2 bg-transparent"
                              style={{ left: `${(x / result.image_width) * 100}%`, top: `${(y / result.image_height) * 100}%`, width: `${(width / result.image_width) * 100}%`, height: `${(height / result.image_height) * 100}%`, borderColor: color, boxShadow: isSelected ? `0 0 0 2px white, 0 0 0 4px ${color}` : undefined }}
                            />
                          )}
                          {showIds && (
                            <button
                              type="button"
                              onClick={(event) => { event.preventDefault(); event.stopPropagation(); selectFromImage(object.id); }}
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
  tone?: 'neutral' | 'green' | 'amber' | 'yellow' | 'red';
  unit?: string;
  compact?: boolean;
}) {
  const colors = {
    neutral: 'border-border bg-surface text-text-primary',
    green: 'border-emerald-200 bg-white text-emerald-600',
    amber: 'border-amber-200 bg-amber-50/70 text-amber-700',
    yellow: 'border-yellow-300 bg-yellow-50 text-yellow-800',
    red: 'border-red-200 bg-red-50 text-red-700',
  }[tone];
  return (
    <div className={`min-w-0 rounded-xl sm:rounded-2xl border p-2.5 sm:p-5 shadow-sm ${colors}`}>
      <p className={`font-semibold uppercase tracking-wide text-text-muted ${compact ? 'text-[9px] sm:text-[11px]' : 'text-[10px] sm:text-xs'}`}>{label}</p>
      <div className="mt-1 flex items-baseline gap-2">
        <p className={`min-w-0 break-words font-bold leading-none ${compact ? 'text-lg sm:text-2xl' : 'text-2xl sm:text-4xl'}`}>{value}</p>
        {unit && <span className="text-[10px] sm:text-xs text-text-muted">{unit}</span>}
      </div>
      <p className={`mt-1.5 sm:mt-2 text-[10px] leading-tight sm:text-xs ${tone === 'neutral' ? 'text-text-muted' : 'opacity-90'}`}>{detail}</p>
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

function objectColor(object: DetectedObject): string {
  if (object.category === 'foreign') return '#dc2626';
  if (object.whole_label === 'whole') return '#2563eb';
  if (object.whole_label === 'broken') return '#eab308';
  return '#6b7280';
}

function pointInPolygon(x: number, y: number, polygon: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const crosses = (yi > y) !== (yj > y)
      && x < ((xj - xi) * (y - yi)) / ((yj - yi) || Number.EPSILON) + xi;
    if (crosses) inside = !inside;
  }
  return inside;
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
    <button type="button" onClick={onClick} aria-pressed={active} className={`inline-flex min-h-10 items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${active ? 'border-brand bg-brand-light text-brand' : 'border-border bg-surface text-text-secondary'}`}>
      {icon}<span>{label}</span>
    </button>
  );
}

function ObjectList({ title, objects, selectedId, onSelect }: { title: string; objects: DetectedObject[]; selectedId: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between text-xs font-medium text-text-secondary mb-2">
        <span>{title}</span><span>{objects.length}</span>
      </div>
      {objects.length ? (
        <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pr-1">
          {objects.map((object) => (
            <button key={object.id} type="button" aria-pressed={object.id === selectedId} onClick={() => onSelect(object.id)} className="min-h-8 min-w-8 rounded-md border px-2 py-1 text-xs font-semibold" style={{ backgroundColor: object.id === selectedId ? objectColor(object) : `${objectColor(object)}18`, color: object.id === selectedId ? '#fff' : objectColor(object), borderColor: `${objectColor(object)}66` }}>
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
