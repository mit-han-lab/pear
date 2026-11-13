import {useState} from 'react';
import {Cable} from 'lucide-react';
import './ConnectModal.css';

type ConnectModalProps = {error: string; onConnect: (token: string) => void};

export function ConnectModal({error, onConnect}: ConnectModalProps) {
  const [token, setToken] = useState('');
  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="modal-icon">
          <Cable size={24} />
        </div>
        <h2>Connect to your station</h2>
        <p>Open the URL that the pear-station gui command prints, or enter this service’s browser token.</p>
        <input aria-label="Browser token" type="password" value={token} onChange={(e) => setToken(e.target.value)} />
        {error && <p className="error-text">{error}</p>}
        <button className="button primary wide" onClick={() => onConnect(token)}>
          Connect
        </button>
      </div>
    </div>
  );
}
