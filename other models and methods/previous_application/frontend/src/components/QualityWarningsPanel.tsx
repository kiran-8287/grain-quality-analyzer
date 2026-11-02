import React, { useState } from 'react';
import { ImageQuality } from '../types';
import { AlertTriangle, Info } from 'lucide-react';

interface QualityWarningsPanelProps {
  quality?: ImageQuality;
  warnings: string[];
  calibrationMode: string;
  pixelsPerMm?: number | null;
  megapixels: number;
}

const PARAMETER_PROVENANCE = [
  ['1. Broken', 'Length vs. estimated whole-kernel reference', 'Robust geometry estimator; Undetermined for N <= 2', 'F — Official concept; image estimator is not the official weight test'],
  ['2. Damaged', 'Binary normal / damaged prediction', 'VGG-19 checkpoint trained on synthetic generated crops; no real rice labels', 'B — Synthetic/demo ML; experimental'],
  ['3. Discoloured', 'CIE76 LAB pixel distance from sample reference', 'Hand-set color heuristic; reference depends on sample', 'C — Computer-vision heuristic'],
  ['4. Chalky', 'No final prediction', 'Synthetic-feature Logistic Regression disabled; returns Undetermined', 'Unavailable — real labeled grain classifier required'],
  ['5. Red', 'LAB and HSV red-area/color rules', 'Hand-set color thresholds; image-based estimate', 'C — Computer-vision heuristic'],
  ['6. Dehusked', 'Brown/bran-like surface coverage', 'RGB/HSV visual estimate; cannot reproduce chemical staining', 'F — Official concept; not directly measurable from image'],
  ['7. Immature / Shrunken', 'Relative breadth, area, solidity and L/B', 'Within-sample geometry rules; not a trained classifier', 'C — Computer-vision heuristic'],
  ['8. Sprouted / Weevilled', 'Normal / sprouted-weevilled prediction', 'ResNet-18 trained on generated crops; score is uncalibrated', 'B — Synthetic/demo ML; experimental'],
  ['9. Foreign Matter', 'Non-rice objects on full image', 'YOLO weights/data unavailable; color/contour heuristic remains active', 'C — Heuristic fallback; detector untrained'],
  ['10. Admixture', 'No lower-class classification', 'Unsupported. Geometry outliers are shown separately and do not establish admixture.', 'Unavailable — labeled lower-class evidence required'],
  ['11. Length', 'Major-axis grain geometry', 'Pixels unless physical scale calibration is supplied', 'D — Mathematical image measurement'],
  ['12. Breadth', 'Minor-axis grain geometry', 'Pixels unless physical scale calibration is supplied', 'D — Mathematical image measurement'],
  ['13. L/B Ratio', 'Length divided by breadth', 'Dimensionless derived geometric measurement', 'D — Derived mathematical measurement'],
  ['14. Total Count', 'Accepted rice grain instances', 'Count after unchanged rice gate and segmentation pipeline', 'C — Computer-vision instance count'],
];

export const QualityWarningsPanel: React.FC<QualityWarningsPanelProps> = ({
  quality,
  warnings,
  calibrationMode,
  pixelsPerMm,
  megapixels,
}) => {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="bg-surface border border-border rounded-2xl p-6 shadow-sm space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <div>
          <span className="text-xs font-semibold text-brand uppercase tracking-wider">Provenance & Quality</span>
          <h3 className="text-lg font-bold text-text-primary mt-0.5">Image Quality & Pipeline Diagnostics</h3>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <div className="text-xs bg-surface-subtle px-3 py-1 rounded-lg border border-border text-text-secondary">
            Resolution: <span className="font-semibold text-text-primary">{megapixels} MP</span> (Informational only)
          </div>
          <button
            type="button"
            aria-expanded={showDetails}
            aria-controls="quality-diagnostics-details"
            onClick={() => setShowDetails((visible) => !visible)}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-border bg-surface text-xs font-medium text-text-secondary hover:bg-surface-subtle hover:text-text-primary transition"
          >
            <Info className="w-3.5 h-3.5" />
            {showDetails ? 'Hide details' : 'Details'}
          </button>
        </div>
      </div>

      <div id="quality-diagnostics-details" hidden={!showDetails} className="space-y-6">
        {/* Warnings List */}
        {warnings.length > 0 && (
          <div className="bg-warning-bg border border-warning/30 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-warning">
              <AlertTriangle className="w-4 h-4" />
              <span>Active Pipeline Warnings ({warnings.length}):</span>
            </div>
            <ul className="list-disc list-inside text-xs text-text-secondary space-y-1">
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {/* 12-MP Rule Scientific Rationale */}
        <div className="p-3.5 bg-surface-subtle border border-border rounded-xl text-xs space-y-1">
          <span className="font-semibold text-text-primary block text-[11px]">
            Engineering Principle: No Fixed 12-Megapixel Gate
          </span>
          <p className="text-text-secondary leading-relaxed text-[11px]">
            Total image megapixels does not equal pixels-per-grain. A 2 MP close-up of 1 grain contains far more detail
            than a 12 MP image with 1,000 tiny grains. The system assesses quality post-segmentation based on median grain pixel area
            ({quality?.median_grain_pixels ?? 0} px/grain) and Laplacian blur variance ({Math.round(quality?.blur_score ?? 0)}).
          </p>
          <p className="text-text-secondary leading-relaxed text-[11px]">
            Engineering thresholds (not calibrated): Laplacian variance GOOD &ge; {quality?.thresholds_used?.blur_laplacian_variance.good_min ?? 100}, FAIR &ge; {quality?.thresholds_used?.blur_laplacian_variance.fair_min ?? 50}, POOR floor {quality?.thresholds_used?.blur_laplacian_variance.poor_min ?? 20}; median grain area GOOD &ge; {quality?.thresholds_used?.median_grain_area_pixels.good_min ?? 400}, FAIR &ge; {quality?.thresholds_used?.median_grain_area_pixels.fair_min ?? 100}, POOR floor {quality?.thresholds_used?.median_grain_area_pixels.poor_min ?? 25} px. Segmentation confidence floor {quality?.thresholds_used?.segmentation_confidence_floor ?? 0.5}; uncertain fraction warning &gt; {Math.round((quality?.thresholds_used?.uncertain_grain_fraction_warning_above ?? 0.2) * 100)}%. No megapixel rejection threshold is used.
          </p>
          <p className="text-text-secondary leading-relaxed text-[11px]">
            Tier score is the equal-weight mean of blur, grain area, uncertain-grain fraction, illumination uniformity and clipping. Score bands: GOOD &ge; 0.80, FAIR &ge; 0.60, otherwise POOR. UNRELIABLE hard failures: blur &lt; {quality?.thresholds_used?.unreliable_hard_failures.blur_below ?? 20}, median grain area &lt; {quality?.thresholds_used?.unreliable_hard_failures.median_grain_area_below_pixels ?? 25} px, mean segmentation confidence &lt; {quality?.thresholds_used?.unreliable_hard_failures.mean_segmentation_confidence_below ?? 0.5}, uncertain fraction &ge; {Math.round((quality?.thresholds_used?.unreliable_hard_failures.uncertain_fraction_at_least ?? 0.5) * 100)}%, low-quality segmentations &ge; {Math.round((quality?.thresholds_used?.unreliable_hard_failures.low_quality_segmentation_fraction_at_least ?? 0.5) * 100)}%, clipping &ge; {Math.round((quality?.thresholds_used?.unreliable_hard_failures.clipped_pixel_fraction_at_least ?? 0.5) * 100)}%, or illumination uniformity &lt; {quality?.thresholds_used?.unreliable_hard_failures.illumination_uniformity_below ?? 0.15}. These engineering criteria require human-rated calibration. Current status: {quality?.tier ?? 'Not assessed'}; reasons: {quality?.unreliable_reasons?.join(' ') || 'none'}
          </p>
        </div>

        {/* Methodology Provenance Matrix */}
        <div>
          <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider mb-2">
            Parameter Provenance & Scientific Basis
          </h4>
          <div className="border border-border rounded-xl overflow-x-auto">
            <div className="min-w-[920px] divide-y divide-border text-xs">
              <div className="grid grid-cols-[1fr_1.3fr_2fr_1.7fr] gap-3 p-2.5 bg-surface-subtle text-[10px] font-semibold text-text-muted uppercase tracking-wider">
                <div>Parameter</div>
                <div>Classification Basis</div>
                <div>Current Method / Evidence</div>
                <div>Provenance Category</div>
              </div>
              {PARAMETER_PROVENANCE.map(([parameter, basis, method, category]) => (
                <div key={parameter} className="grid grid-cols-[1fr_1.3fr_2fr_1.7fr] gap-3 p-2.5 text-text-secondary">
                  <span className="font-medium text-text-primary">{parameter}</span>
                  <span>{basis}</span>
                  <span>{method}</span>
                  <span>{category}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
