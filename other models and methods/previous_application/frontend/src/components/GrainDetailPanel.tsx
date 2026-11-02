import React from 'react';
import { GrainInstance } from '../types';
import { Info, CheckCircle2, AlertTriangle, HelpCircle } from 'lucide-react';

interface GrainDetailPanelProps {
  grain: GrainInstance | null;
  unit: string;
  showGrainId?: boolean;
}

export const GrainDetailPanel: React.FC<GrainDetailPanelProps> = ({ grain, unit, showGrainId = true }) => {
  if (!grain) {
    return (
      <div className="bg-surface border border-border rounded-2xl p-6 text-center text-text-muted">
        <Info className="w-8 h-8 mx-auto text-text-muted mb-2" />
        <h3 className="text-sm font-semibold text-text-primary">No Grain Selected</h3>
        <p className="text-xs text-text-secondary mt-1">
          Select any grain from the image or grain table to inspect its mask-based geometric measurements and Whole vs Broken classification.
        </p>
      </div>
    );
  }

  const { geometry } = grain;
  const brokenDefect = grain.defects?.broken;
  const status = brokenDefect?.broken_label;
  const ratio = brokenDefect?.length_ratio ?? brokenDefect?.broken_ratio;
  const refLen = brokenDefect?.whole_kernel_length_ref;
  const reason = brokenDefect?.classification_reason;
  const method = brokenDefect?.method;

  const effectiveLen =
    brokenDefect?.effective_length ??
    (geometry.effective_length_mm ??
      geometry.length_mm ??
      geometry.effective_length_pixels ??
      geometry.length_pixels);

  const lengthDisplay = geometry.length_mm !== null && geometry.length_mm !== undefined
    ? `${geometry.length_mm} mm`
    : `${geometry.length_pixels} px`;

  const breadthDisplay = geometry.breadth_mm !== null && geometry.breadth_mm !== undefined
    ? `${geometry.breadth_mm} mm`
    : `${geometry.breadth_pixels} px`;

  return (
    <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <div>
          <span className="text-xs uppercase font-semibold text-brand tracking-wider">Per-Grain Inspector</span>
          <h2 className="text-xl font-bold text-text-primary">
            {showGrainId ? `Grain #${grain.id}` : 'Grain'}
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`text-xs px-2.5 py-1 rounded-full font-medium border ${
              grain.segmentation_quality === 'good'
                ? 'bg-success-bg text-success border-success/20'
                : 'bg-warning-bg text-warning border-warning/20'
            }`}
          >
            Quality: {grain.segmentation_quality}
          </span>
          {grain.is_touching && (
            <span className="text-xs px-2.5 py-1 rounded-full font-medium bg-error-bg text-error border border-error/20">
              Touching / Overlapping
            </span>
          )}
        </div>
      </div>

      {/* Whole vs Broken Prominent Classification Card */}
      <div
        className={`p-3.5 rounded-xl border ${
          status === 'whole'
            ? 'bg-emerald-50/60 border-emerald-200'
            : status === 'broken'
            ? 'bg-rose-50/60 border-rose-200'
            : 'bg-slate-50 border-slate-200'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {status === 'whole' && <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
            {status === 'broken' && <AlertTriangle className="w-5 h-5 text-rose-600" />}
            {status !== 'whole' && status !== 'broken' && <HelpCircle className="w-5 h-5 text-slate-500" />}
            <span className="text-xs font-bold uppercase tracking-wider text-text-secondary">
              Classification Status
            </span>
          </div>
          <span
            className={`px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider border ${
              status === 'whole'
                ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                : status === 'broken'
                ? 'bg-rose-100 text-rose-800 border-rose-300'
                : 'bg-slate-100 text-slate-700 border-slate-300'
            }`}
          >
            {status || 'undetermined'}
          </span>
        </div>

        <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <div className="bg-white/80 p-2 rounded-lg border border-border/60">
            <span className="text-text-muted block text-[10px] uppercase font-semibold">Effective Length</span>
            <span className="font-bold text-text-primary text-sm">{effectiveLen} {unit}</span>
          </div>
          <div className="bg-white/80 p-2 rounded-lg border border-border/60">
            <span className="text-text-muted block text-[10px] uppercase font-semibold">Reference Length</span>
            <span className="font-bold text-text-primary text-sm">{refLen ? `${refLen} ${unit}` : 'N/A'}</span>
          </div>
          <div className="bg-white/80 p-2 rounded-lg border border-border/60">
            <span className="text-text-muted block text-[10px] uppercase font-semibold">Length Ratio</span>
            <span className="font-bold text-text-primary text-sm">{ratio != null ? `${(ratio * 100).toFixed(1)}%` : 'N/A'}</span>
          </div>
          <div className="bg-white/80 p-2 rounded-lg border border-border/60">
            <span className="text-text-muted block text-[10px] uppercase font-semibold">Threshold</span>
            <span className="font-bold text-text-primary text-sm">75.0% (3/4 rule)</span>
          </div>
        </div>

        {reason && (
          <p className="mt-2 text-[11px] text-text-secondary">
            <span className="font-semibold text-text-primary">Basis:</span> {reason} (method: {method || 'unknown'})
          </p>
        )}
      </div>

      {/* Geometry Metrics Grid */}
      <div>
        <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider mb-2">Geometric Measurements</h4>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Ellipse Length</span>
            <span className="font-bold text-text-primary text-sm">{lengthDisplay}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Breadth</span>
            <span className="font-bold text-text-primary text-sm">{breadthDisplay}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">L/B Ratio</span>
            <span className="font-bold text-text-primary text-sm">{geometry.lb_ratio ?? 'N/A'}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Mask Area</span>
            <span className="font-bold text-text-primary text-sm">{geometry.area_pixels.toLocaleString()} px²</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Solidity</span>
            <span className="font-bold text-text-primary text-sm">{geometry.solidity.toFixed(4)}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Confidence</span>
            <span className="font-bold text-success text-sm">{Math.round(grain.confidence * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Segmentation Metadata */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
          <span className="text-text-muted block text-[11px]">Bounding Box</span>
          <span className="font-mono text-text-primary text-[11px]">
            [{grain.bbox.map((v) => Math.round(v)).join(', ')}]
          </span>
        </div>
        <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
          <span className="text-text-muted block text-[11px]">Centroid</span>
          <span className="font-mono text-text-primary text-[11px]">
            ({grain.centroid.map((v) => Math.round(v)).join(', ')})
          </span>
        </div>
      </div>
    </div>
  );
};
