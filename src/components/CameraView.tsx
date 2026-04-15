import React, { useEffect, useRef } from 'react';
import { CAMERA_MAX_RETRIES, CAMERA_RETRY_DELAY_MS } from '../config';

interface CameraViewProps {
  onVideoReady?: (videoElement: HTMLVideoElement) => void;
  onError?: (message: string) => void;
  isProcessing: boolean;
}

function errorMessage(err: unknown): { msg: string; retryable: boolean } {
  if (err instanceof DOMException) {
    switch (err.name) {
      case 'NotAllowedError':
        return { msg: 'Camera access denied. Please enable camera in your device settings.', retryable: false };
      case 'NotFoundError':
        return { msg: 'No camera found on this device.', retryable: false };
      case 'NotReadableError':
        return { msg: 'Camera is in use by another app.', retryable: true };
      case 'OverconstrainedError':
        return { msg: 'Camera does not support the requested settings.', retryable: true };
    }
  }
  return { msg: 'Camera error. Retrying...', retryable: true };
}

const CameraView: React.FC<CameraViewProps> = ({ onVideoReady, onError, isProcessing }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let cancelled = false;

    const startCamera = async (attempt: number) => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            if (onVideoReady && videoRef.current) {
              onVideoReady(videoRef.current);
            }
          };
        }
      } catch (err) {
        console.error('Camera error:', err);
        const { msg, retryable } = errorMessage(err);

        if (retryable && attempt < CAMERA_MAX_RETRIES && !cancelled) {
          onError?.(`${msg} Retrying... (${attempt}/${CAMERA_MAX_RETRIES})`);
          await new Promise(r => setTimeout(r, CAMERA_RETRY_DELAY_MS));
          if (!cancelled) await startCamera(attempt + 1);
        } else {
          const finalMsg = attempt >= CAMERA_MAX_RETRIES
            ? 'Camera unavailable after multiple attempts. Please restart the app.'
            : msg;
          onError?.(finalMsg);
        }
      }
    };

    startCamera(1);

    return () => {
      cancelled = true;
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  return (
    <div className="camera-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`camera-feed ${isProcessing ? 'dimmed' : ''}`}
      />
    </div>
  );
};

export default CameraView;
