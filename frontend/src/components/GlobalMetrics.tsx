import React from 'react';
import { SampleSummary } from '../types';

interface GlobalMetricsProps {
  summary: SampleSummary;
  warnings: string[];
}

export const GlobalMetrics: React.FC<GlobalMetricsProps> = ({ summary, warnings }) => {
  const formatValue = (value: number | null | undefined, decimals = 2) =>
    value == null ? 'N/A' : `${Number(value).toFixed(decimals)}`;

  const totalGrains = summary.total_count ?? summary.total_rice_grains ?? 0;
  const brokenGrains = summary.broken_count ?? 0;
  const wholeGrains = summary.whole_count ?? Math.max(0, totalGrains - brokenGrains);
  const undeterminedGrains = summary.undetermined_count ?? 0;
  const foreignObjects = summary.foreign_matter_count ?? 0;
  const brokenPct = summary.broken_percent != null ? `${summary.broken_percent.toFixed(2)}%` : 'N/A';
  const wholePct = summary.whole_percent != null ? `${summary.whole_percent.toFixed(2)}%` : 'N/A';

  const referenceSourceBadge = () => {
    const src = summary.reference_source;
    const refLen = summary.whole_reference_length;
    const unit = summary.measurement_unit || 'px';
    const dataStatus = summary.reference_data_status;
    const profileName = summary.reference_profile_name;

    if (src === 'profile') {
      const isPixelUnit = unit === 'px' || unit === 'pixels' || unit.toLowerCase().startsWith('pixel');
      const statusColor = dataStatus === 'Measured' ? 'text-emerald-700 bg-emerald-50 border-emerald-200' : 'text-amber-700 bg-amber-50 border-amber-200';
      return (
        <div className="flex flex-col items-end gap-0.5">
          <span className={`text-[11px] font-semibold ${statusColor} border px-2 py-0.5 rounded-md`}>
            Reference: Profile{profileName ? ` (${profileName})` : ''}
          </span>
          <span className="text-[10px] text-text-secondary">
            Whole-kernel reference: {refLen ? `${refLen} ${unit}` : 'Configured'}
            {' · '}
            Criterion: &lt; 75% of reference = Broken
          </span>
          {isPixelUnit && (
            <span className="text-[10px] italic text-text-muted">
              Pixel reference is scale-specific — not a universal physical measurement.
            </span>
          )}
          {dataStatus && (
            <span className="text-[10px] text-text-muted">
              Data status: {dataStatus}
            </span>
          )}
        </div>
      );
    }
    if (src === 'sample_derived') {
      return (
        <div className="flex flex-col items-end gap-0.5">
          <span className="text-[11px] font-semibold text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded-md">
            Sample-Derived Reference
          </span>
          <span className="text-[10px] text-text-secondary">
            Whole-kernel reference: {refLen ? `${refLen} ${unit}` : 'Candidate population'}
          </span>
          {dataStatus && (
            <span className="text-[10px] text-text-muted">
              Data status: {dataStatus}
            </span>
          )}
        </div>
      );
    }
    return (
      <div className="flex flex-col items-end gap-0.5">
        <span className="text-[11px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-md">
          Reference: Not Determinable
        </span>
        <span className="text-[10px] text-text-secondary">
          No valid whole-kernel reference available for this image.
        </span>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Header Badge */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-widest text-brand font-bold bg-brand-light border border-brand/20 px-2.5 py-1 rounded-full">
            Whole vs Broken Analysis
          </span>
          <span className="text-xs text-text-secondary font-medium">
            3/4 Whole-Kernel Length Rule (0.75× Reference)
          </span>
          <span className="text-[10px] text-amber-700 font-medium bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
            Foreign objects excluded from grain counts
          </span>
        </div>
        <div>{referenceSourceBadge()}</div>
      </div>

      {/* Regulatory note */}
      <div className="text-[10px] text-text-muted bg-surface-subtle border border-border/50 rounded-lg px-3 py-2">
        Image-based result is calculated by grain count. Regulatory broken-rice limits may use
        mass-based measurements and are not established by count alone.
      </div>

      {/* Primary Whole vs Broken Summary Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Total Count */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-bold text-text-muted uppercase tracking-wider block">Total Grains</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-3xl font-black text-text-primary">{totalGrains}</span>
            <span className="text-xs text-text-muted">grains</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">
            {summary.uncertain_grains > 0 ? `${summary.uncertain_grains} touching/merged` : 'Clean segmentation'}
          </span>
        </div>

        {/* Whole Count */}
        <div className="bg-surface border border-emerald-200 bg-emerald-50/30 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-bold text-emerald-800 uppercase tracking-wider block">Whole Grains</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-3xl font-black text-emerald-600">{wholeGrains}</span>
            <span className="text-xs text-emerald-700 font-medium">≥ 75% length</span>
          </div>
          <span className="text-[10px] text-emerald-700 mt-1 block">
            {wholePct !== 'N/A' ? `${wholePct} of sample` : 'Intact kernels'}
          </span>
        </div>

        {/* Broken Count */}
        <div className="bg-surface border border-rose-200 bg-rose-50/30 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-bold text-rose-800 uppercase tracking-wider block">Broken Grains</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-3xl font-black text-rose-600">{brokenGrains}</span>
            <span className="text-xs text-rose-700 font-medium">&lt; 75% length</span>
          </div>
          <span className="text-[10px] text-rose-700 mt-1 block">
            {undeterminedGrains > 0 ? `${undeterminedGrains} undetermined` : 'Pieces & fragments'}
          </span>
        </div>

        {/* Foreign Objects */}
        <div className="bg-surface border border-amber-200 bg-amber-50/30 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-bold text-amber-800 uppercase tracking-wider block">Foreign Objects</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-3xl font-black text-amber-600">{foreignObjects}</span>
            <span className="text-xs text-amber-700 font-medium">non-rice</span>
          </div>
          <span className="text-[10px] text-amber-700 mt-1 block">
            {foreignObjects > 0 ? 'Detected in image' : 'No foreign matter'}
          </span>
        </div>

        {/* Whole Percentage */}
        <div className="bg-surface border border-emerald-200 bg-emerald-50/30 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-bold text-emerald-800 uppercase tracking-wider block">Whole %</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-3xl font-black text-emerald-600">{wholePct}</span>
          </div>
          <span className="text-[10px] text-emerald-700 mt-1 block">
            {wholeGrains > 0 ? 'Intact kernels' : 'No whole grains'}
          </span>
        </div>

        {/* Broken Percentage */}
        <div className="bg-surface border border-rose-200 bg-rose-50/30 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-bold text-rose-800 uppercase tracking-wider block">Broken % by count</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-3xl font-black text-rose-600">{brokenPct}</span>
          </div>
          <span className="text-[10px] text-rose-700 mt-1 block">
            {brokenGrains > 0 ? 'Fragments & pieces' : 'No broken grains'}
          </span>
        </div>
      </div>

      {/* Secondary Geometry Overview */}
      <div className="grid grid-cols-3 gap-3">
        {/* Average Length */}
        <div className="bg-surface-subtle border border-border/70 rounded-lg p-2.5">
          <span className="text-[10px] font-semibold text-text-muted uppercase block">Avg Length</span>
          <div className="mt-0.5 flex items-baseline gap-1">
            <span className="text-base font-bold text-text-primary">{formatValue(summary.average_length)}</span>
            <span className="text-[10px] text-text-muted font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[9px] text-text-muted">Median: {formatValue(summary.median_length)}</span>
        </div>

        {/* Average Breadth */}
        <div className="bg-surface-subtle border border-border/70 rounded-lg p-2.5">
          <span className="text-[10px] font-semibold text-text-muted uppercase block">Avg Breadth</span>
          <div className="mt-0.5 flex items-baseline gap-1">
            <span className="text-base font-bold text-text-primary">{formatValue(summary.average_breadth)}</span>
            <span className="text-[10px] text-text-muted font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[9px] text-text-muted">Median: {formatValue(summary.median_breadth)}</span>
        </div>

        {/* Average L/B Ratio */}
        <div className="bg-surface-subtle border border-border/70 rounded-lg p-2.5">
          <span className="text-[10px] font-semibold text-text-muted uppercase block">Avg L/B Ratio</span>
          <div className="mt-0.5 flex items-baseline gap-1">
            <span className="text-base font-bold text-text-primary">{formatValue(summary.average_lb_ratio)}</span>
          </div>
          <span className="text-[9px] text-text-muted">Median: {formatValue(summary.median_lb_ratio)}</span>
        </div>
      </div>
    </div>
  );
};
