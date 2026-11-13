import {X} from 'lucide-react';
import {Camera} from './Camera';
import type {ServerState} from './protocol';

type CameraModalProps = {
  role: string;
  label: string;
  state: ServerState | null;
  connected: boolean;
  frame?: string;
  onClose: () => void;
};

export function CameraModal({role, label, state, connected, frame, onClose}: CameraModalProps) {
  return (
    <div className="modal-backdrop camera-modal" onClick={onClose}>
      <button aria-label="Close camera" className="modal-close">
        <X size={22} />
      </button>
      <Camera role={role} label={label} state={state} connected={connected} frame={frame} large />
    </div>
  );
}
