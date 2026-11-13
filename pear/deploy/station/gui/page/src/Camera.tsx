import {useState} from 'react';
import {Cable, Maximize2, Video} from 'lucide-react';
import type {ServerState} from './protocol';
import './Camera.css';

export const CAMERA_FRESH_AGE_S = 0.5; // a frame older than this counts as no signal

type CameraProps = {
  role: string;
  label: string;
  state: ServerState | null;
  connected: boolean;
  frame?: string;
  large?: boolean;
  onExpand?: (role: string) => void;
};

export function Camera({role, label, state, connected, frame, large = false, onExpand}: CameraProps) {
  const info = state?.cameras.roles[role];
  const cameraError = state?.cameras.errors ? state.cameras.errors[role] : state?.cameras.error;
  const fresh = connected && info && info.age < CAMERA_FRESH_AGE_S && !cameraError;
  const simulated = state?.backend === 'fake';
  const [size, setSize] = useState(''); // the latest frame's, which is the camera's own resolution
  const footer = [size, info && `${Math.round(info.age * 1000)} ms old`].filter(Boolean).join(' · ');
  return (
    <div className={`camera ${large ? 'large' : ''}`}>
      <div className="camera-head">
        <span>
          <Video size={15} />
          {label}
        </span>
        <span className={`camera-live ${fresh ? '' : 'stale'}`}>
          <i />
          {fresh ? (simulated ? 'SIMULATED' : 'LIVE') : 'NO SIGNAL'}
        </span>
      </div>
      <div className="camera-image">
        {frame && (
          <img
            src={frame}
            alt={`${label} ${simulated ? 'simulation' : 'camera'} feed`}
            onLoad={(e) => setSize(`${e.currentTarget.naturalWidth} × ${e.currentTarget.naturalHeight}`)}
          />
        )}
        <div className="camera-cross" />
        {!fresh && (
          <div className="signal-overlay">
            <Cable size={26} />
            <span>{cameraError || 'Waiting for camera'}</span>
          </div>
        )}
        <div className="camera-footer">
          <span>{footer}</span>
          <button title={`Expand ${label}`} aria-label={`Expand ${label}`} onClick={() => onExpand?.(role)}>
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
