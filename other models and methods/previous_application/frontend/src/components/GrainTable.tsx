import React from 'react';
import { GrainInstance } from '../types';

interface GrainTableProps {
  grains: GrainInstance[];
  selectedGrainId: number | null;
  onSelectGrain: (id: number) => void;
  unit: string;
  showGrainIds?: boolean;
}

const statusBadgeClass = (status: string | undefined) => {
  switch (status) {
    case 'whole':
      return 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold';
    case 'broken':
      return 'bg-rose-100 text-rose-800 border-rose-300 font-bold';
    case 'undetermined':
      return 'bg-slate-100 text-slate-700 border-slate-300 font-medium';
    default:
      return 'bg-surface-subtle text-text-muted border border-border';
  }
};

export const GrainTable: React.FC<GrainTableProps> = ({
  grains,
  selectedGrainId,
  onSelectGrain,
  unit,
  showGrainIds = true,
}) => {
  return (
    <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs font-semibold text-brand uppercase tracking-wider">Per-Grain Table</span>
          <h3 className="text-base font-bold text-text-primary">Grain Population Overview</h3>
        </div>
        <span className="text-xs text-text-muted">{grains.length} accepted instances</span>
      </div>

      <div className="overflow-x-auto max-h-72 overflow-y-auto border border-border rounded-xl">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="sticky top-0 bg-surface-subtle text-text-secondary font-semibold uppercase text-[10px] tracking-wider z-10">
            <tr>
              <th className="py-2 px-3">Grain #</th>
              <th className="py-2 px-3">Status</th>
              <th className="py-2 px-3">Effective Length ({unit})</th>
              <th className="py-2 px-3">Length Ratio</th>
              <th className="py-2 px-3">Breadth ({unit})</th>
              <th className="py-2 px-3">L/B Ratio</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {grains.map((g) => {
              const isSelected = g.id === selectedGrainId;
              const brokenDefect = g.defects?.broken;
              const status = brokenDefect?.broken_label;
              const ratio = brokenDefect?.length_ratio ?? brokenDefect?.broken_ratio;

              const len =
                brokenDefect?.effective_length ??
                (g.geometry.effective_length_mm ??
                  g.geometry.length_mm ??
                  g.geometry.effective_length_pixels ??
                  g.geometry.length_pixels);

              const brd =
                g.geometry.breadth_mm !== null && g.geometry.breadth_mm !== undefined
                  ? g.geometry.breadth_mm
                  : g.geometry.breadth_pixels;

              return (
                <tr
                  key={g.id}
                  onClick={() => onSelectGrain(g.id)}
                  className={`cursor-pointer transition ${
                    isSelected
                      ? 'bg-brand-light border-l-4 border-brand'
                      : 'hover:bg-surface-subtle'
                  }`}
                >
                  <td className="py-2.5 px-3 font-semibold text-text-primary">
                    {showGrainIds ? `#${g.id}` : '—'}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] uppercase tracking-wide border ${statusBadgeClass(
                        status
                      )}`}
                    >
                      {status || 'undetermined'}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-semibold text-text-primary">{len}</td>
                  <td className="py-2.5 px-3 font-mono text-text-secondary">
                    {ratio != null ? `${(ratio * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-text-secondary">{brd}</td>
                  <td className="py-2.5 px-3 font-mono text-brand font-medium">
                    {g.geometry.lb_ratio ?? 'N/A'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
