import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Crosshair, Eye, EyeOff, Circle, Square } from 'lucide-react';
import { GrainInstance, ForeignObject } from '../types';

interface AnnotatedViewerProps {
  annotatedImageUrl?: string;
  originalImageUrl?: string;
  grains: GrainInstance[];
  selectedGrainId: number | null;
  onSelectGrain: (id: number) => void;
  foreignMatter?: ForeignObject[];
  showMasks?: boolean;
  showBoxes?: boolean;
  showIds?: boolean;
  onToggleMasks?: (v: boolean) => void;
  onToggleBoxes?: (v: boolean) => void;
  onToggleIds?: (v: boolean) => void;
}

function colorForGrain(grain: GrainInstance, alpha = 0.35): string {
  const status = grain.defects?.broken?.broken_label;
  if (status === 'whole') {
    return `hsla(142, 72%, 45%, ${alpha})`;
  }
  if (status === 'broken') {
    return `hsla(0, 84%, 55%, ${alpha})`;
  }
  return `hsla(215, 25%, 60%, ${alpha})`;
}

function strokeColorForGrain(grain: GrainInstance): string {
  const status = grain.defects?.broken?.broken_label;
  if (status === 'whole') {
    return 'hsl(142, 76%, 36%)';
  }
  if (status === 'broken') {
    return 'hsl(0, 84%, 50%)';
  }
  return 'hsl(215, 25%, 50%)';
}

export const AnnotatedViewer: React.FC<AnnotatedViewerProps> = ({
  annotatedImageUrl,
  originalImageUrl,
  grains,
  selectedGrainId,
  onSelectGrain,
  foreignMatter = [],
  showMasks = true,
  showBoxes = true,
  showIds = true,
  onToggleMasks,
  onToggleBoxes,
  onToggleIds,
}) => {
  const [zoom, setZoom] = useState(1);
  const [hoveredGrainId, setHoveredGrainId] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const useCanvas = !!originalImageUrl;

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgRef.current) return;

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
      for (const grain of grains) {
        const polygon = grain.mask_polygon;
        if (!polygon || polygon.length < 3) continue;
        const isSelected = grain.id === selectedGrainId;
        const isHovered = grain.id === hoveredGrainId;
        const alpha = isSelected ? 0.60 : isHovered ? 0.50 : 0.38;

        ctx.beginPath();
        polygon.forEach(([px, py], i) => {
          const x = px * sx;
          const y = py * sy;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = colorForGrain(grain, alpha);
        ctx.fill();
      }
    }

    if (showBoxes) {
      for (const grain of grains) {
        const [x, y, w, h] = grain.bbox;
        const isSelected = grain.id === selectedGrainId;
        const isHovered = grain.id === hoveredGrainId;
        ctx.strokeStyle = strokeColorForGrain(grain);
        ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 1.5;
        ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
      }
    }

    if (showMasks) {
      for (const grain of grains) {
        if (grain.id !== selectedGrainId) continue;
        const polygon = grain.mask_polygon;
        if (!polygon || polygon.length < 3) continue;

        ctx.beginPath();
        polygon.forEach(([px, py], i) => {
          const x = px * sx;
          const y = py * sy;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.strokeStyle = strokeColorForGrain(grain);
        ctx.lineWidth = 4;
        ctx.stroke();
      }
    }

    if (foreignMatter.length) {
      for (const fm of foreignMatter) {
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

    if (showIds) {
      for (const grain of grains) {
        const [cx, cy] = grain.centroid;
        const px = cx * sx;
        const py = cy * sy;
        const label = `#${grain.id}`;
        const confText = `${Math.round(grain.confidence * 100)}%`;

        ctx.font = 'bold 11px Inter, sans-serif';
        const idMetrics = ctx.measureText(label);
        const confMetrics = ctx.measureText(confText);
        const idW = idMetrics.width + 8;
        const confW = confMetrics.width + 8;
        const totalW = idW + confW + 2;
        const badgeH = 18;
        const badgeX = px - totalW / 2;
        const badgeY = py - badgeH / 2;

        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = colorForGrain(grain, 0.9);
        ctx.lineWidth = 1;
        ctx.beginPath();
        const r = Math.min(5, totalW / 2, badgeH / 2);
        ctx.moveTo(badgeX + r, badgeY);
        ctx.lineTo(badgeX + totalW - r, badgeY);
        ctx.quadraticCurveTo(badgeX + totalW, badgeY, badgeX + totalW, badgeY + r);
        ctx.lineTo(badgeX + totalW, badgeY + badgeH - r);
        ctx.quadraticCurveTo(badgeX + totalW, badgeY + badgeH, badgeX + totalW - r, badgeY + badgeH);
        ctx.lineTo(badgeX + r, badgeY + badgeH);
        ctx.quadraticCurveTo(badgeX, badgeY + badgeH, badgeX, badgeY + badgeH - r);
        ctx.lineTo(badgeX, badgeY + r);
        ctx.quadraticCurveTo(badgeX, badgeY, badgeX + r, badgeY);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        let cursorX = badgeX + 4;
        ctx.fillStyle = '#1F2937';
        ctx.fillText(label, cursorX, badgeY + 13);
        cursorX += idW;

        ctx.fillStyle = strokeColorForGrain(grain);
        ctx.fillRect(cursorX, badgeY + 3, 1, badgeH - 6);
        cursorX += 2;

        ctx.fillStyle = strokeColorForGrain(grain);
        ctx.fillText(confText, cursorX, badgeY + 13);
      }
    }
  }, [grains, selectedGrainId, hoveredGrainId, foreignMatter, showMasks, showBoxes, showIds]);

  useEffect(() => {
    if (!originalImageUrl) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgRef.current = img;
      requestAnimationFrame(drawCanvas);
    };
    img.src = originalImageUrl;
  }, [originalImageUrl, drawCanvas]);

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
    if (!canvas || !imgRef.current) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const displayWidth = rect.width;
    const displayHeight = rect.height;
    const sx = imgRef.current.naturalWidth / displayWidth;
    const sy = imgRef.current.naturalHeight / displayHeight;
    const ix = px * sx;
    const iy = py * sy;

    for (let i = grains.length - 1; i >= 0; i--) {
      const g = grains[i];
      const [bx, by, bw, bh] = g.bbox;
      if (ix >= bx && ix <= bx + bw && iy >= by && iy <= by + bh) {
        const polygon = g.mask_polygon;
        if (polygon && polygon.length >= 3) {
          if (pointInPolygon([ix, iy], polygon)) return g.id;
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
    if (id !== null) onSelectGrain(id);
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

  const hoveredGrain = grains.find((g) => g.id === hoveredGrainId) || null;

  const toggleButton = (active: boolean, onToggle: ((v: boolean) => void) | undefined, label: string, IconOn: React.FC<{ className?: string }>, IconOff: React.FC<{ className?: string }>) => {
    if (!onToggle) return null;
    return (
      <button
        onClick={() => onToggle(!active)}
        className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium transition-all border ${
          active ? 'bg-brand text-white border-brand shadow-sm' : 'bg-surface text-text-secondary border-border hover:bg-surface-subtle hover:text-text-primary'
        }`}
        title={active ? `Hide ${label}` : `Show ${label}`}
      >
        {active ? <IconOn className="w-3 h-3" /> : <IconOff className="w-3 h-3" />}
        <span className="hidden sm:inline">{label}</span>
        <span className={`text-[9px] font-bold ${active ? 'text-white/80' : 'text-text-muted'}`}>{active ? 'ON' : 'OFF'}</span>
      </button>
    );
  };

  return (
    <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm flex flex-col">
      {/* Controls Bar */}
      <div className="px-4 py-3 border-b border-border flex flex-wrap items-center justify-between gap-3 bg-surface-subtle">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-text-primary">Annotated Inspection</span>
          <span className="text-[10px] bg-surface border border-border text-text-muted px-2 py-0.5 rounded-full">
            {grains.length} Grains Mapped
          </span>
          <div className="flex items-center gap-1.5 ml-1">
            <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-800 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Whole (≥75%)
            </span>
            <span className="inline-flex items-center gap-1 text-[10px] font-bold text-rose-800 bg-rose-50 border border-rose-200 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span> Broken (&lt;75%)
            </span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Grain Selector Dropdown */}
          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <Crosshair className="w-3.5 h-3.5 text-brand" />
            <select
              value={selectedGrainId ?? ''}
              onChange={(e) => onSelectGrain(Number(e.target.value))}
              className="bg-surface border border-border rounded-md px-2 py-1 text-text-primary text-xs focus:outline-none focus:border-brand"
            >
              <option value="">Select Grain #ID...</option>
              {grains.map((g) => (
                <option key={g.id} value={g.id}>
                  Grain #{g.id} {g.is_touching ? '(Touching)' : ''}
                </option>
              ))}
            </select>
          </div>

          {toggleButton(showIds, onToggleIds, 'IDs', Eye, EyeOff)}
          {toggleButton(showMasks, onToggleMasks, 'Masks', Circle, Circle)}
          {toggleButton(showBoxes, onToggleBoxes, 'Boxes', Square, Square)}

          {/* Zoom controls */}
          <div className="flex items-center bg-surface rounded-lg p-0.5 border border-border">
            <button
              onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
              className="p-1 hover:text-text-primary text-text-muted transition"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] px-1.5 text-text-secondary font-mono">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
              className="p-1 hover:text-text-primary text-text-muted transition"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(1)}
              className="p-1 hover:text-text-primary text-text-muted transition ml-0.5"
              title="Reset Zoom"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Image Canvas Container */}
      <div ref={containerRef} className="relative overflow-auto p-4 flex items-center justify-center min-h-[380px] bg-surface-subtle">
        <div
          style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
          className="transition-transform duration-200 w-full"
        >
          {useCanvas ? (
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onMouseMove={handleCanvasMove}
              onMouseLeave={handleCanvasLeave}
              className="block w-full h-auto rounded-lg shadow-sm border border-border cursor-crosshair"
              style={{ imageRendering: 'auto' }}
            />
          ) : (
            <img
              src={annotatedImageUrl}
              alt="Annotated Rice Grains"
              className="block w-full max-w-full h-auto object-contain rounded-lg shadow-sm border border-border"
            />
          )}

          {/* Tooltip */}
          {tooltipPos && hoveredGrain && useCanvas && (
            <div
              className="pointer-events-none absolute z-20 bg-surface border border-border rounded-lg shadow-xl px-3 py-2 text-xs"
              style={{ left: tooltipPos.x, top: tooltipPos.y }}
            >
              <div className="flex items-center justify-between gap-3 mb-1">
                <span className="font-semibold text-text-primary">Grain #{hoveredGrain.id}</span>
                <span
                  className={`text-[9px] font-black uppercase px-1.5 py-0.5 rounded border ${
                    hoveredGrain.defects?.broken?.broken_label === 'whole'
                      ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                      : hoveredGrain.defects?.broken?.broken_label === 'broken'
                      ? 'bg-rose-100 text-rose-800 border-rose-300'
                      : 'bg-slate-100 text-slate-700 border-slate-300'
                  }`}
                >
                  {hoveredGrain.defects?.broken?.broken_label || 'undetermined'}
                </span>
              </div>
              <p className="text-text-muted">
                Length:{' '}
                <span className="font-medium text-text-primary">
                  {hoveredGrain.defects?.broken?.effective_length ??
                    hoveredGrain.geometry.effective_length_pixels ??
                    hoveredGrain.geometry.length_pixels}{' '}
                  px
                </span>
              </p>
              {hoveredGrain.defects?.broken?.length_ratio != null && (
                <p className="text-text-muted">
                  Ratio:{' '}
                  <span className="font-medium text-text-primary">
                    {(hoveredGrain.defects.broken.length_ratio * 100).toFixed(1)}%
                  </span>
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer Instructions */}
      <div className="px-4 py-2 border-t border-border bg-surface-subtle text-[11px] text-text-muted flex items-center justify-between">
        <span>
          {useCanvas
            ? 'Click a grain to inspect it. Use dropdown or table to select a grain.'
            : 'Grains tagged with colored instance masks and #ID tags.'}
        </span>
        <span>Use dropdown or table to select a grain</span>
      </div>
    </div>
  );
};
