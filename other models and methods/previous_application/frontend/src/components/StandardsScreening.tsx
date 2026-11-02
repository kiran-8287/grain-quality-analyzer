import React from 'react';
import { StandardsResult } from '../types';
import { AlertCircle, HelpCircle, FileText, CheckCircle2, XCircle } from 'lucide-react';

interface StandardsScreeningProps {
  standards: StandardsResult;
}

export const StandardsScreening: React.FC<StandardsScreeningProps> = ({ standards }) => {
  const { screening, official_grade, standard_reference } = standards;

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'WITHIN REFERENCE LIMIT':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-success-bg text-success border border-success/20">
            <CheckCircle2 className="w-3 h-3" />
            Image estimate within reference
          </span>
        );
      case 'EXCEEDS REFERENCE LIMIT':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-error-bg text-error border border-error/20">
            <XCircle className="w-3 h-3" />
            Image estimate above reference
          </span>
        );
      case 'NOT DETERMINABLE':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-surface-subtle text-text-muted border border-border">
            <HelpCircle className="w-3 h-3" />
            Not determinable
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-surface-subtle text-text-muted border border-border">
            <HelpCircle className="w-3 h-3" />
            Not Assessable
          </span>
        );
    }
  };

  return (
    <div className="bg-surface border border-border rounded-2xl p-6 shadow-sm space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-border">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-brand uppercase tracking-wider">Official Standards Screening</span>
            <span className="text-xs bg-brand-light text-brand border border-border px-2 py-0.5 rounded-full">
              {standard_reference?.display_label || 'Reference standard; season not verified'}
            </span>
          </div>
          <h3 className="text-lg font-bold text-text-primary mt-1">Image-Based Standard Screening</h3>
          <p className="text-xs text-text-secondary">
            Image-based observed fractions are compared with reference limits only; official weight-based compliance is not established.
          </p>
        </div>
      </div>

      {/* Screening Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-border text-text-muted font-semibold uppercase text-[10px] tracking-wider">
              <th className="py-2.5 px-3">Quality Parameter</th>
              <th className="py-2.5 px-3">Observed Value</th>
              <th className="py-2.5 px-3">Reference Limit</th>
              <th className="py-2.5 px-3">Status</th>
              <th className="py-2.5 px-3">Measurement Basis</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {Object.entries(screening).map(([key, item]) => {
              const formattedName = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
              return (
                <tr key={key} className="hover:bg-surface-subtle transition">
                  <td className="py-3 px-3 font-medium text-text-primary">{formattedName}</td>
                  <td className="py-3 px-3 font-semibold text-text-primary">{item.observed_value}</td>
                  <td className="py-3 px-3 text-text-secondary font-mono">{item.reference_limit}</td>
                  <td className="py-3 px-3">{getStatusBadge(item.status)}</td>
                  <td className="py-3 px-3 text-[11px] text-text-muted">{item.basis}</td>
                </tr>
              );
            })}


          </tbody>
        </table>
      </div>


    </div>
  );
};
