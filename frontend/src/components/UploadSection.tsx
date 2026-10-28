import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileUp, Camera, CameraOff, RefreshCw, X } from 'lucide-react';

interface UploadSectionProps {
  onAnalyze: (file: File) => void;
  isLoading: boolean;
}

type CameraState = 'idle' | 'requesting' | 'active' | 'captured' | 'error';

export const UploadSection: React.FC<UploadSectionProps> = ({ onAnalyze, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [capturedPreviewUrl, setCapturedPreviewUrl] = useState<string | null>(null);
  const [cameraSupported, setCameraSupported] = useState<boolean>(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setSelectedFile(file);
    setPreviewUrl(null);
    setCapturedPreviewUrl(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewUrl(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const startCamera = async () => {
    setCameraError(null);
    setCapturedPreviewUrl(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    setCameraState('requesting');
    if (fileInputRef.current) fileInputRef.current.value = '';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraSupported(false);
      setCameraState('error');
      setCameraError('Camera is not supported in this browser.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraState('active');
    } catch (err: any) {
      stopCamera();
      setCameraState('error');
      setCameraError(
        err?.name === 'NotAllowedError'
          ? 'Camera permission was denied. Please allow camera access and try again.'
          : err?.name === 'NotFoundError'
          ? 'No camera was found on this device.'
          : 'Unable to access the camera.'
      );
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    setCapturedPreviewUrl(dataUrl);
    setCameraState('captured');
    stopCamera();
  };

  const retakePhoto = async () => {
    setCapturedPreviewUrl(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    await startCamera();
  };

  const cancelCamera = () => {
    stopCamera();
    setCapturedPreviewUrl(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    setCameraState('idle');
    setCameraError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const analyzeCaptured = () => {
    const canvas = canvasRef.current;
    if (!canvas || !capturedPreviewUrl) return;

    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
      setSelectedFile(file);
      onAnalyze(file);
    }, 'image/jpeg', 0.92);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;
    onAnalyze(selectedFile);
  };

  const resetAll = () => {
    stopCamera();
    setSelectedFile(null);
    setPreviewUrl(null);
    setCapturedPreviewUrl(null);
    setCameraState('idle');
    setCameraError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isCameraActive = cameraState === 'active' || cameraState === 'requesting';

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Intro Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-text-primary tracking-tight sm:text-4xl">
          Rice Grain Auto-Label
        </h1>
        <p className="text-sm text-text-secondary mt-2 max-w-2xl mx-auto">
          Upload an image or take a photo to create a masked image and count rice grains and foreign matter.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Upload / Camera Entry Cards */}
        {!isCameraActive && cameraState !== 'captured' && !previewUrl && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Upload Card */}
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
                dragActive
                  ? 'border-brand bg-brand-light'
                  : 'border-border bg-surface hover:border-border-dark hover:bg-surface-subtle'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.tiff,.tif"
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
              />
              <div className="w-14 h-14 rounded-2xl bg-brand-light border border-border flex items-center justify-center text-brand mx-auto mb-3">
                <Upload className="w-7 h-7" />
              </div>
              <div className="text-sm font-semibold text-text-primary">Upload Image</div>
              <p className="text-xs text-text-muted mt-1 max-w-[220px] mx-auto">
                Choose a photo from your device
              </p>
            </div>

            {/* Camera Card */}
            <button
              type="button"
              onClick={startCamera}
              disabled={isLoading}
              className="relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 border-border bg-surface hover:border-border-dark hover:bg-surface-subtle disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <div className="w-14 h-14 rounded-2xl bg-brand-light border border-border flex items-center justify-center text-brand mx-auto mb-3">
                <Camera className="w-7 h-7" />
              </div>
              <div className="text-sm font-semibold text-text-primary">Use Camera</div>
              <p className="text-xs text-text-muted mt-1 max-w-[220px] mx-auto">
                Take a photo directly
              </p>
            </button>
          </div>
        )}

        {/* Camera Error */}
        {cameraState === 'error' && cameraError && (
          <div className="bg-error-bg border border-error/30 rounded-xl p-4 flex items-start gap-3">
            <CameraOff className="w-5 h-5 text-error shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold text-error text-sm">Camera Unavailable</span>
              <p className="text-xs text-text-secondary mt-1">{cameraError}</p>
            </div>
            <button
              type="button"
              onClick={() => { setCameraState('idle'); setCameraError(null); }}
              className="text-xs text-error hover:text-text-primary underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Camera Preview */}
        {isCameraActive && (
          <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm">
            <div className="px-4 py-3 border-b border-border bg-surface-subtle flex items-center justify-between">
              <span className="text-xs font-semibold text-text-primary">Camera Preview</span>
              <button
                type="button"
                onClick={cancelCamera}
                className="flex items-center gap-1.5 text-xs font-medium text-text-secondary hover:text-error transition"
              >
                <X className="w-3.5 h-3.5" />
                Cancel
              </button>
            </div>
            <div className="relative bg-black flex items-center justify-center min-h-[300px] sm:min-h-[420px]">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-auto max-h-[60vh] object-contain"
              />
            </div>
            <div className="px-4 py-3 border-t border-border bg-surface-subtle flex justify-center">
              <button
                type="button"
                onClick={capturePhoto}
                disabled={cameraState === 'requesting'}
                className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-sm bg-brand hover:bg-brand-dark text-white transition shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Camera className="w-4 h-4" />
                Capture Photo
              </button>
            </div>
          </div>
        )}

        {/* Captured Preview */}
        {cameraState === 'captured' && capturedPreviewUrl && (
          <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm">
            <div className="px-4 py-3 border-b border-border bg-surface-subtle">
              <span className="text-xs font-semibold text-text-primary">Captured Photo</span>
            </div>
            <div className="p-4 flex items-center justify-center bg-surface-subtle">
              <img
                src={capturedPreviewUrl}
                alt="Captured"
                className="max-h-72 object-contain rounded-lg border border-border"
              />
            </div>
            <div className="px-4 py-3 border-t border-border bg-surface-subtle flex flex-wrap items-center justify-center gap-3">
              <button
                type="button"
                onClick={retakePhoto}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm bg-surface hover:bg-surface-subtle text-text-secondary border border-border transition"
              >
                <RefreshCw className="w-4 h-4" />
                Retake
              </button>
              <button
                type="button"
                onClick={analyzeCaptured}
                disabled={isLoading}
                className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-sm bg-brand hover:bg-brand-dark text-white transition shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <FileUp className="w-4 h-4" />
                <span>{isLoading ? 'Analyzing...' : 'Analyze Photo'}</span>
              </button>
            </div>
          </div>
        )}

        {/* Upload Drop Zone & Preview (shown when file selected or preview exists) */}
        {!isCameraActive && cameraState !== 'captured' && (previewUrl || selectedFile) && (
          <div
            className="relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-200 border-border bg-surface hover:border-border-dark hover:bg-surface-subtle"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.tiff,.tif"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <div className="flex flex-col items-center gap-4">
              <div className="relative group max-w-sm rounded-xl overflow-hidden border border-border bg-surface-subtle">
                <img src={previewUrl || ''} alt="Upload Preview" className="max-h-72 object-contain mx-auto" />
                <div className="absolute inset-0 bg-text-primary/0 group-hover:bg-text-primary/10 transition-opacity flex items-center justify-center">
                  <span className="text-xs text-text-primary bg-surface px-3 py-1.5 rounded-lg border border-border">
                    Click or drop to replace
                  </span>
                </div>
              </div>
              <div className="text-xs text-text-secondary">
                <span className="font-semibold text-text-primary">{selectedFile?.name}</span> ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
              </div>
            </div>
          </div>
        )}

        {/* Submit / Reset */}
        <div className="flex flex-wrap items-center justify-center gap-3">
          {(previewUrl || selectedFile || cameraState === 'captured') && (
            <button
              type="button"
              onClick={resetAll}
              className="px-5 py-2.5 rounded-xl font-semibold text-sm bg-surface hover:bg-surface-subtle text-text-secondary border border-border transition"
            >
              Clear
            </button>
          )}
          <button
            type="submit"
            disabled={!selectedFile || isLoading}
            className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-200 ${
              !selectedFile || isLoading
                ? 'bg-surface-subtle text-text-muted cursor-not-allowed border border-border'
                : 'bg-brand hover:bg-brand-dark text-white shadow-sm'
            }`}
          >
            <FileUp className="w-4 h-4" />
            <span>{isLoading ? 'Processing Image...' : 'Run Auto-Label'}</span>
          </button>
        </div>
      </form>

      {/* Hidden capture canvas */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};
