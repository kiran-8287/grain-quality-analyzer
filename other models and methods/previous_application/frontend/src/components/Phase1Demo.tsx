import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  Play,
  X,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  Image as ImageIcon,
  CheckCircle2,
  XCircle,
  Download,
  Sparkles,
} from 'lucide-react';
import {
  Phase1AnalysisResponse,
  Phase1GrainInstance,
  Phase1ForeignMatterInstance,
} from '../types';

const CONFIDENCE_THRESHOLDS = { HIGH: 0.85, MEDIUM: 0.6 };

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

function getConfidenceLabel(confidence: number): 'HIGH' | 'MEDIUM' | 'LOW' {
  if (confidence >= CONFIDENCE_THRESHOLDS.HIGH) return 'HIGH';
  if (confidence >= CONFIDENCE_THRESHOLDS.MEDIUM) return 'MEDIUM';
  return 'LOW';
}

function hueForId(id: number): number {
  return (id * 47) % 360;
}

function colorForGrain(id: number, alpha = 0.35): string {
  return `hsla(${hueForId(id)}, 65%, 50%, ${alpha})`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

const EMPTY_RESULT: Phase1AnalysisResponse = {
  success: false,
  rice_detected: false,
  rice_count: 0,
  foreign_matter_count: 0,
  unresolved_cluster_count: 0,
  grains: [],
  foreign_matter: [],
  unresolved_clusters: [],
  processing: { inference_ms: 0, total_ms: 0, tiling_used: false, num_tiles: 0 },
  method: '',
  model_version: '',
};

export const Phase1Demo: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Phase1AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedGrainId, setSelectedGrainId] = useState<number | null>(null);
  const [hoveredGrainId, setHoveredGrainId] = useState<number | null>(null);
  const [showMasks, setShowMasks] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showIds, setShowIds] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const selectedGrain: Phase1GrainInstance | null =
    result?.grains?.find((g) => g.id === selectedGrainId) ?? null;

  const hoveredGrain: Phase1GrainInstance | null =
    result?.grains?.find((g) => g.id === hoveredGrainId) ?? null;

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setSelectedGrainId(null);
    setErrorMessage(null);
  }, [previewUrl]);

  const handleClear = useCallback(() => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setSelectedGrainId(null);
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [previewUrl]);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setIsLoading(true);
    setErrorMessage(null);
    setSelectedGrainId(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const endpoints = [`${API_BASE}/api/phase1/analyze`, `${API_BASE}/api/analyze`];
      let response: Response | null = null;
      let lastError: any = null;

      for (const endpoint of endpoints) {
        try {
          response = await fetch(endpoint, { method: 'POST', body: formData });
          if (response.ok) break;
          lastError = new Error(`Server error (${response.status})`);
        } catch (err) {
          lastError = err;
          response = null;
        }
      }

      if (!response || !response.ok) {
        if (lastError) throw lastError;
        throw new Error('Failed to reach analysis endpoint');
      }

      const data: Phase1AnalysisResponse = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Analysis returned unsuccessful');
      }

      const grainsWithLabels = data.grains.map((g) => ({
        ...g,
        confidence_label: g.confidence_label || getConfidenceLabel(g.confidence),
      }));

      setResult({ ...data, grains: grainsWithLabels });
      if (grainsWithLabels.length > 0) {
        setSelectedGrainId(grainsWithLabels[0].id);
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'An unexpected error occurred during analysis.');
      setResult(EMPTY_RESULT);
    } finally {
      setIsLoading(false);
    }
  }, [file]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result || !imgRef.current) return;

    const img = imgRef.current;
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvas.clientWidth;
    const aspectRatio = img.naturalHeight / img.naturalWidth;
    const displayHeight = Math.round(displayWidth * aspectRatio);

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.height = `${displayHeight}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, displayWidth, displayHeight);
    ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

    const sx = displayWidth / img.naturalWidth;
    const sy = displayHeight / img.naturalHeight;

    if (showMasks) {
      for (const grain of result.grains) {
        if (!grain.mask_polygon || grain.mask_polygon.length < 3) continue;
        const isSelected = grain.id === selectedGrainId;
        const isHovered = grain.id === hoveredGrainId;
        const alpha = isSelected ? 0.55 : isHovered ? 0.48 : 0.35;

        ctx.beginPath();
        grain.mask_polygon.forEach(([px, py], i) => {
          const x = px * sx;
          const y = py * sy;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = colorForGrain(grain.id, alpha);
        ctx.fill();
      }
    }

    if (showBoxes) {
      for (const grain of result.grains) {
        const [x, y, w, h] = grain.bbox;
        const isSelected = grain.id === selectedGrainId;
        const isHovered = grain.id === hoveredGrainId;
        ctx.strokeStyle = colorForGrain(grain.id, 1);
        ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 1.5;
        ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
      }
    }

    if (showMasks) {
      for (const grain of result.grains) {
        if (grain.id !== selectedGrainId) continue;
        if (!grain.mask_polygon || grain.mask_polygon.length < 3) continue;

        ctx.beginPath();
        grain.mask_polygon.forEach(([px, py], i) => {
          const x = px * sx;
          const y = py * sy;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.strokeStyle = `hsl(${hueForId(grain.id)}, 80%, 55%)`;
        ctx.lineWidth = 4;
        ctx.stroke();
      }
    }

    if (result.foreign_matter && showBoxes) {
      for (const fm of result.foreign_matter) {
        const [x, y, w, h] = fm.bbox;
        ctx.save();
        ctx.strokeStyle = '#DC2626';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
        ctx.restore();

        ctx.save();
        ctx.fillStyle = '#DC2626';
        const labelText = `${fm.class} ${Math.round(fm.confidence * 100)}%`;
        ctx.font = 'bold 11px Inter, sans-serif';
        const textMetrics = ctx.measureText(labelText);
        const labelW = textMetrics.width + 10;
        const labelH = 18;
        const labelX = x * sx;
        const labelY = Math.max(0, y * sy - labelH - 2);
        ctx.fillRect(labelX, labelY, labelW, labelH);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(labelText, labelX + 5, labelY + 13);
        ctx.restore();
      }
    }

    for (const grain of result.grains) {
      const [cx, cy] = grain.centroid;
      const px = cx * sx;
      const py = cy * sy;
      const label = `#${grain.id}`;
      const showIdHere = showIds;
      const showConfHere = showConfidence;

      if (!showIdHere && !showConfHere) continue;

      ctx.font = 'bold 11px Inter, sans-serif';
      const idMetrics = ctx.measureText(label);
      const confText = `${Math.round(grain.confidence * 100)}%`;
      const confMetrics = ctx.measureText(confText);
      const idW = idMetrics.width + 8;
      const confW = confMetrics.width + 8;
      const totalW = showIdHere && showConfHere ? idW + confW + 2 : Math.max(idW, confW);
      const badgeH = 18;
      const badgeX = px - totalW / 2;
      const badgeY = py - badgeH / 2;

      ctx.fillStyle = '#FFFFFF';
      ctx.strokeStyle = colorForGrain(grain.id, 0.9);
      ctx.lineWidth = 1;
      thisRoundRect(ctx, badgeX, badgeY, totalW, badgeH, 5);
      ctx.fill();
      ctx.stroke();

      let cursorX = badgeX + 4;
      if (showIdHere) {
        ctx.fillStyle = '#1F2937';
        ctx.fillText(label, cursorX, badgeY + 13);
        cursorX += idW;
      }
      if (showIdHere && showConfHere) {
        ctx.fillStyle = colorForGrain(grain.id, 1);
        ctx.fillRect(cursorX, badgeY + 3, 1, badgeH - 6);
        cursorX += 2;
      }
      if (showConfHere) {
        ctx.fillStyle = colorForGrain(grain.id, 1);
        ctx.fillText(confText, cursorX, badgeY + 13);
      }
    }
  }, [result, showMasks, showBoxes, showIds, showConfidence, selectedGrainId, hoveredGrainId]);

  function thisRoundRect(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number, r: number
  ) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  useEffect(() => {
    if (!previewUrl) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgRef.current = img;
      requestAnimationFrame(drawCanvas);
    };
    img.src = previewUrl;
  }, [previewUrl, result, drawCanvas]);

  useEffect(() => {
    if (imgRef.current) {
      requestAnimationFrame(drawCanvas);
    }
  }, [drawCanvas]);

  useEffect(() => {
    const onResize = () => {
      if (imgRef.current) requestAnimationFrame(drawCanvas);
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [drawCanvas]);

  const findGrainAt = (clientX: number, clientY: number): number | null => {
    const canvas = canvasRef.current;
    if (!canvas || !result || !imgRef.current) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const displayWidth = rect.width;
    const displayHeight = rect.height;
    const sx = imgRef.current.naturalWidth / displayWidth;
    const sy = imgRef.current.naturalHeight / displayHeight;
    const ix = px * sx;
    const iy = py * sy;

    for (let i = result.grains.length - 1; i >= 0; i--) {
      const g = result.grains[i];
      const [bx, by, bw, bh] = g.bbox;
      if (ix >= bx && ix <= bx + bw && iy >= by && iy <= by + bh) {
        if (g.mask_polygon && g.mask_polygon.length >= 3) {
          if (pointInPolygon([ix, iy], g.mask_polygon)) return g.id;
        } else {
          return g.id;
        }
      }
    }
    return null;
  };

  function pointInPolygon(pt: [number, number], poly: Array<[number, number]>): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i];
      const [xj, yj] = poly[j];
      const intersect =
        yi > pt[1] !== yj > pt[1] &&
        pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi + 1e-9) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const id = findGrainAt(e.clientX, e.clientY);
    setSelectedGrainId(id);
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const id = findGrainAt(e.clientX, e.clientY);
    setHoveredGrainId(id);
    const canvas = canvasRef.current;
    if (id && canvas) {
      const rect = canvas.getBoundingClientRect();
      setTooltipPos({ x: e.clientX - rect.left + 14, y: e.clientY - rect.top + 14 });
    } else {
      setTooltipPos(null);
    }
  };

  const handleCanvasLeave = () => {
    setHoveredGrainId(null);
    setTooltipPos(null);
  };

  const confidenceBadgeClass = (label: 'HIGH' | 'MEDIUM' | 'LOW') => {
    switch (label) {
      case 'HIGH': return 'bg-success-bg text-success-text border-success/30';
      case 'MEDIUM': return 'bg-warning-bg text-warning-text border-warning/30';
      case 'LOW': return 'bg-orange-50 text-orange-700 border-orange-200';
    }
  };

  const toggleButton = (active: boolean, onToggle: () => void, label: string, iconOn: React.ReactNode, iconOff: React.ReactNode) => (
    <button
      onClick={onToggle}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
        active
          ? 'bg-brand text-white border-brand shadow-sm'
          : 'bg-surface text-text-secondary border-border hover:bg-surface-subtle hover:text-text-primary'
      }`}
    >
      {active ? iconOn : iconOff}
      {label}
      <span className={`text-[10px] font-bold ${active ? 'text-white/80' : 'text-text-muted'}`}>
        {active ? 'ON' : 'OFF'}
      </span>
    </button>
  );

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="bg-gradient-to-r from-brand to-sage rounded-2xl p-6 text-white shadow-md relative overflow-hidden">
        <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle at 20% 50%, #fff 0%, transparent 40%), radial-gradient(circle at 80% 80%, #fff 0%, transparent 35%)' }} />
        <div className="relative z-10 flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-white/15 backdrop-blur-sm border border-white/20 flex items-center justify-center text-3xl">
            🌾
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.2em] text-white/70 font-semibold mb-1">
              Instance Segmentation Demo
            </p>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight">
              GRAIN QUALITY ANALYZER — PHASE 1
            </h1>
            <p className="text-sm text-white/80 mt-1 max-w-xl">
              Upload an image to detect, segment, and count individual rice grains.
              Interactive inspection with masks, bounding boxes, and per-grain metadata.
            </p>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {errorMessage && (
        <div className="bg-error-bg border border-error/30 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-error shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-error text-sm">Analysis Failed</span>
            <p className="text-xs text-text-secondary mt-1">{errorMessage}</p>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-xs text-error hover:text-text-primary underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Upload Section */}
      <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm">
        <div className="flex flex-wrap items-center gap-4">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm bg-surface-subtle border-2 border-dashed border-border-dark hover:border-brand hover:bg-brand-light hover:text-brand transition-all"
          >
            <Upload className="w-4 h-4" />
            Choose Image
          </button>

          {file && (
            <div className="flex items-center gap-3 px-4 py-2 bg-surface-subtle border border-border rounded-xl flex-1 min-w-0">
              <ImageIcon className="w-4 h-4 text-brand shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-text-primary truncate">{file.name}</p>
                <p className="text-xs text-text-muted">{formatBytes(file.size)}</p>
              </div>
            </div>
          )}

          <div className="flex items-center gap-2 ml-auto">
            <button
              onClick={handleAnalyze}
              disabled={!file || isLoading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm bg-brand text-white hover:bg-brand-dark disabled:bg-sage/40 disabled:text-white/60 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Analyze
                </>
              )}
            </button>

            <button
              onClick={handleClear}
              disabled={!file && !result}
              className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl font-medium text-sm bg-surface border border-border text-text-secondary hover:text-error hover:border-error/40 hover:bg-error-bg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <X className="w-4 h-4" />
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="bg-surface border border-border rounded-2xl p-16 text-center shadow-sm">
          <div className="w-16 h-16 mx-auto rounded-2xl bg-brand-light flex items-center justify-center mb-4">
            <Loader2 className="w-8 h-8 text-brand animate-spin" />
          </div>
          <p className="text-lg font-semibold text-text-primary mb-1">Analyzing image...</p>
          <p className="text-sm text-text-secondary">
            Running instance segmentation, detecting rice grains and foreign matter.
          </p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !result && (
        <div className="bg-surface border-2 border-dashed border-border-dark rounded-2xl p-12 sm:p-16 text-center shadow-sm">
          <div className="w-20 h-20 mx-auto rounded-3xl bg-brand-light flex items-center justify-center mb-5">
            <Sparkles className="w-10 h-10 text-brand" />
          </div>
          <h2 className="text-xl font-bold text-text-primary mb-2">Ready to Analyze</h2>
          <p className="text-sm text-text-secondary max-w-md mx-auto mb-6 leading-relaxed">
            Upload a rice grain image to see Phase 1 instance segmentation in action. Each grain
            will be individually detected, outlined, and labeled with its confidence score.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-2xl mx-auto text-left">
            {[
              { t: '1. Upload', d: 'Click "Choose Image" to select a rice sample photo' },
              { t: '2. Analyze', d: 'Click "Analyze" to run the segmentation pipeline' },
              { t: '3. Inspect', d: 'Click grains on the canvas to view their details' },
            ].map((s) => (
              <div key={s.t} className="bg-surface-subtle border border-border rounded-xl p-4">
                <p className="text-xs font-bold text-brand uppercase tracking-wide mb-1">{s.t}</p>
                <p className="text-xs text-text-secondary leading-relaxed">{s.d}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results: Two Panel Layout */}
      {!isLoading && result && result.success && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Panel - Canvas (60%) */}
          <div className="lg:col-span-3 space-y-4">
            <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm">
              {/* Toggles */}
              <div className="px-4 py-3 border-b border-border bg-surface-subtle flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-text-primary">Segmentation Overlay</span>
                  <span className="text-[10px] bg-surface border border-border text-text-muted px-2 py-0.5 rounded-full">
                    {result.grains.length} Grains
                  </span>
                  {result.foreign_matter_count > 0 && (
                    <span className="text-[10px] bg-error-bg border border-error/30 text-error-text px-2 py-0.5 rounded-full font-medium">
                      {result.foreign_matter_count} FM
                    </span>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {toggleButton(showMasks, () => setShowMasks((v) => !v), 'Masks', <Eye className="w-3.5 h-3.5" />, <EyeOff className="w-3.5 h-3.5" />)}
                  {toggleButton(showBoxes, () => setShowBoxes((v) => !v), 'Boxes', <Eye className="w-3.5 h-3.5" />, <EyeOff className="w-3.5 h-3.5" />)}
                  {toggleButton(showIds, () => setShowIds((v) => !v), 'IDs', <Eye className="w-3.5 h-3.5" />, <EyeOff className="w-3.5 h-3.5" />)}
                  {toggleButton(showConfidence, () => setShowConfidence((v) => !v), 'Conf.', <Eye className="w-3.5 h-3.5" />, <EyeOff className="w-3.5 h-3.5" />)}
                </div>
              </div>

              {/* Canvas */}
              <div ref={containerRef} className="relative bg-surface-subtle p-3 overflow-auto" style={{ minHeight: 400 }}>
                {previewUrl ? (
                  <div className="relative inline-block w-full">
                    <canvas
                      ref={canvasRef}
                      onClick={handleCanvasClick}
                      onMouseMove={handleCanvasMove}
                      onMouseLeave={handleCanvasLeave}
                      className="block w-full h-auto rounded-lg shadow-sm border border-border cursor-crosshair"
                      style={{ imageRendering: 'auto' }}
                    />

                    {/* Tooltip */}
                    {tooltipPos && hoveredGrain && (
                      <div
                        className="phase1-tooltip pointer-events-none absolute z-20 bg-surface border border-border rounded-lg shadow-xl px-3 py-2 text-xs"
                        style={{ left: tooltipPos.x, top: tooltipPos.y }}
                      >
                        <p className="font-semibold text-text-primary mb-0.5">Grain #{hoveredGrain.id}</p>
                        <p className="text-text-muted">
                          Confidence: <span className="font-medium text-text-primary">{(hoveredGrain.confidence * 100).toFixed(1)}%</span>
                        </p>
                        <p className="text-text-muted">
                          Area: <span className="font-medium text-text-primary">{hoveredGrain.area_pixels.toLocaleString()} px²</span>
                        </p>
                      </div>
                    )}

                    {/* Selected grain pulse ring */}
                    {selectedGrain && canvasRef.current && (
                      <SelectedGrainPulse
                        grain={selectedGrain}
                        canvas={canvasRef.current}
                        imgNaturalWidth={imgRef.current?.naturalWidth || 0}
                        imgNaturalHeight={imgRef.current?.naturalHeight || 0}
                      />
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-[400px] text-text-muted text-sm">
                    No image preview available
                  </div>
                )}
              </div>

              <div className="px-4 py-2 border-t border-border bg-surface-subtle text-[11px] text-text-muted flex items-center justify-between">
                <span>Click a grain to inspect it. Toggle overlays above.</span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm border-2 border-dashed border-error" /> Foreign Matter
                </span>
              </div>
            </div>
          </div>

          {/* Right Panel - Summary & Details (40%) */}
          <div className="lg:col-span-2 space-y-4">
            {/* Summary Cards */}
            <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-bold text-text-primary">Detection Summary</h2>
                <span className="text-[10px] uppercase tracking-wide text-text-muted font-semibold">
                  Phase 1
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className={`rounded-xl p-3 border ${result.rice_detected ? 'bg-success-bg border-success/20' : 'bg-error-bg border-error/20'}`}>
                  <p className="text-[10px] uppercase tracking-wide font-semibold mb-1.5" style={{ color: result.rice_detected ? '#2D5A3F' : '#7A2E28' }}>
                    Rice Detected
                  </p>
                  <div className="flex items-center gap-1.5">
                    {result.rice_detected ? (
                      <CheckCircle2 className="w-5 h-5 text-success" />
                    ) : (
                      <XCircle className="w-5 h-5 text-error" />
                    )}
                    <span className="text-lg font-bold" style={{ color: result.rice_detected ? '#2D5A3F' : '#7A2E28' }}>
                      {result.rice_detected ? 'YES' : 'NO'}
                    </span>
                  </div>
                </div>

                <div className="bg-brand-light/50 border border-brand/20 rounded-xl p-3">
                  <p className="text-[10px] uppercase tracking-wide text-brand-dark font-semibold mb-1.5">
                    Rice Count
                  </p>
                  <p className="text-2xl font-bold text-brand leading-none">
                    {result.rice_count}
                  </p>
                  <p className="text-[10px] text-brand/70 mt-1">individual grains</p>
                </div>

                <div className="bg-error-bg/60 border border-error/20 rounded-xl p-3">
                  <p className="text-[10px] uppercase tracking-wide font-semibold mb-1.5 text-error-text">
                    Foreign Matter
                  </p>
                  <p className="text-xl font-bold text-error">
                    {result.foreign_matter_count}
                  </p>
                </div>

                <div className="bg-warning-bg/60 border border-warning/20 rounded-xl p-3">
                  <p className="text-[10px] uppercase tracking-wide font-semibold mb-1.5 text-warning-text">
                    Unresolved Clusters
                  </p>
                  <p className="text-xl font-bold text-warning">
                    {result.unresolved_cluster_count}
                  </p>
                </div>
              </div>

              {/* Processing Stats */}
              <div className="mt-4 pt-4 border-t border-border space-y-2">
                <p className="text-[10px] uppercase tracking-wide text-text-muted font-semibold">
                  Processing
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Inference</span>
                    <span className="font-mono font-semibold text-text-primary">{result.processing.inference_ms} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Total</span>
                    <span className="font-mono font-semibold text-text-primary">{result.processing.total_ms} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Method</span>
                    <span className="font-mono text-text-primary text-right truncate max-w-[130px]" title={result.method}>{result.method || '—'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Tiling</span>
                    <span className="font-mono font-semibold text-text-primary">
                      {result.processing.tiling_used ? `${result.processing.num_tiles} tiles` : 'Off'}
                    </span>
                  </div>
                </div>
                {result.model_version && (
                  <p className="text-[11px] font-mono text-text-muted pt-1 border-t border-border/60 truncate">
                    Model: {result.model_version}
                  </p>
                )}
              </div>
            </div>

            {/* Selected Grain Detail */}
            <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm">
              {selectedGrain ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-bold text-text-primary flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: `hsl(${hueForId(selectedGrain.id)}, 65%, 50%)` }}
                      />
                      Grain #{selectedGrain.id}
                    </h2>
                    <span className={`text-[10px] font-bold uppercase tracking-wide px-2 py-1 rounded-md border ${confidenceBadgeClass(selectedGrain.confidence_label)}`}>
                      {selectedGrain.confidence_label}
                    </span>
                  </div>

                  <div className="space-y-3 text-sm">
                    <div className="flex items-center justify-between py-2 border-b border-border/60">
                      <span className="text-text-secondary">Confidence</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-surface-subtle rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${selectedGrain.confidence * 100}%`,
                              backgroundColor: `hsl(${hueForId(selectedGrain.id)}, 65%, 50%)`,
                            }}
                          />
                        </div>
                        <span className="font-mono font-semibold text-text-primary w-14 text-right">
                          {(selectedGrain.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between py-2 border-b border-border/60">
                      <span className="text-text-secondary">BBox</span>
                      <span className="font-mono text-xs text-text-primary bg-surface-subtle px-2 py-1 rounded-md">
                        [{selectedGrain.bbox.map((v) => Math.round(v)).join(', ')}]
                      </span>
                    </div>

                    <div className="flex items-center justify-between py-2 border-b border-border/60">
                      <span className="text-text-secondary">Area</span>
                      <span className="font-mono font-semibold text-text-primary">
                        {selectedGrain.area_pixels.toLocaleString()} px²
                      </span>
                    </div>

                    <div className="flex items-center justify-between py-2 border-b border-border/60">
                      <span className="text-text-secondary">Centroid</span>
                      <span className="font-mono text-xs text-text-primary bg-surface-subtle px-2 py-1 rounded-md">
                        ({selectedGrain.centroid.map((v) => Math.round(v)).join(', ')})
                      </span>
                    </div>

                    <div className="flex items-center justify-between py-2 border-b border-border/60">
                      <span className="text-text-secondary">Touching</span>
                      <span className={`text-xs font-bold px-2.5 py-1 rounded-md ${selectedGrain.is_touching ? 'bg-warning-bg text-warning-text' : 'bg-success-bg text-success-text'}`}>
                        {selectedGrain.is_touching ? 'Yes' : 'No'}
                      </span>
                    </div>

                    <div className="flex items-center justify-between py-2">
                      <span className="text-text-secondary">Method</span>
                      <span className="font-mono text-xs text-text-primary bg-surface-subtle px-2 py-1 rounded-md max-w-[180px] truncate" title={selectedGrain.segmentation_method}>
                        {selectedGrain.segmentation_method || '—'}
                      </span>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-border/60">
                    <button
                      disabled
                      title="Crop endpoint not yet available in backend"
                      className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-medium text-sm bg-surface-subtle border border-border text-text-muted cursor-not-allowed"
                    >
                      <Download className="w-4 h-4" />
                      Download Crop
                      <span className="text-[10px] bg-border/60 text-text-muted px-2 py-0.5 rounded-full ml-1">
                        Soon
                      </span>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center py-10">
                  <div className="w-12 h-12 mx-auto rounded-xl bg-surface-subtle border border-border flex items-center justify-center mb-3">
                    <CrosshairIcon />
                  </div>
                  <p className="text-sm font-medium text-text-primary mb-1">No Grain Selected</p>
                  <p className="text-xs text-text-muted max-w-xs mx-auto leading-relaxed">
                    Click any grain on the left canvas to view its segmentation details.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Result but no success */}
      {!isLoading && result && !result.success && (
        <div className="bg-surface border border-border rounded-2xl p-10 text-center shadow-sm">
          <XCircle className="w-12 h-12 text-error mx-auto mb-3" />
          <p className="text-lg font-semibold text-text-primary mb-1">No Analysis Result</p>
          <p className="text-sm text-text-secondary">
            Upload a new image and click "Analyze" to run Phase 1 segmentation.
          </p>
        </div>
      )}
    </div>
  );
};

const CrosshairIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8A948E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="22" y1="12" x2="18" y2="12" />
    <line x1="6" y1="12" x2="2" y2="12" />
    <line x1="12" y1="6" x2="12" y2="2" />
    <line x1="12" y1="22" x2="12" y2="18" />
  </svg>
);

interface PulseProps {
  grain: Phase1GrainInstance;
  canvas: HTMLCanvasElement;
  imgNaturalWidth: number;
  imgNaturalHeight: number;
}

const SelectedGrainPulse: React.FC<PulseProps> = ({ grain, canvas, imgNaturalWidth, imgNaturalHeight }) => {
  const [, forceRerender] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => forceRerender((v) => v + 1));
    return () => cancelAnimationFrame(id);
  });

  if (!imgNaturalWidth || !imgNaturalHeight) return null;
  const rect = canvas.getBoundingClientRect();
  const sx = rect.width / imgNaturalWidth;
  const sy = rect.height / imgNaturalHeight;
  const [x, y, w, h] = grain.bbox;
  const inset = 6;

  return (
    <div
      className="phase1-pulse-ring pointer-events-none absolute rounded-xl"
      style={{
        left: x * sx - inset,
        top: y * sy - inset,
        width: w * sx + inset * 2,
        height: h * sy + inset * 2,
        border: `3px solid hsl(${hueForId(grain.id)}, 80%, 55%)`,
      }}
    />
  );
};
