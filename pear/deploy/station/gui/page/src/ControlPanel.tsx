import {Eye, Home, LoaderCircle, Power, RotateCcw, Settings2} from 'lucide-react';
import {Api} from './api';
import {PerformancePanel} from './PerformancePanel';
import type {ServerState} from './protocol';
import type {ContextUsage} from './turns';
import './ControlPanel.css';

const INFERENCE_PHASES = ['Thinking', 'Compacting'];

type ControlPanelProps = {
  state: ServerState | null;
  context: ContextUsage;
  connected: boolean;
  mine: boolean;
  pending: string;
  camerasReady: boolean;
  command: (name: string, data?: Record<string, unknown>) => Promise<boolean>;
  onHome: (openGrippers: boolean) => void;
  onDisable: () => void;
  onReconnect: () => void;
  onError: (message: string) => void;
};

/** The robot status and the buttons that move it or change the model. */
export function ControlPanel({
  state,
  context,
  connected,
  mine,
  pending,
  camerasReady,
  command,
  onHome,
  onDisable,
  onReconnect,
  onError,
}: ControlPanelProps) {
  const enabled = state?.robot.status !== 'OFF' && !!state;
  const ready = state?.robot.status === 'READY';
  const fault = state?.robot.status === 'FAULT';
  const busy = !!state?.busy;
  const task = state?.task;
  const stopped = !!state?.stop_latched;
  const recovering = state?.phase === 'Recovering control';
  const keepGrippers = !!state?.profile.home_keep_grippers;
  const homeDisabled = !mine || !enabled || fault || stopped || !!pending;

  function statusMessage(): string {
    if (!connected) return 'Connection lost. Actual robot state is unknown.';
    if (recovering) {
      return `${state?.robot.recovery?.phase || 'Checking hardware'}. Recovering the hold at the current pose. Your task stays paused.`;
    }
    if (fault) {
      if (state?.error?.includes('Control recovery failed:')) return state.error;
      return 'Recover control checks hardware and gradually resets the hold to the current pose. The task stays paused; gripper targets stay unchanged.';
    }
    if (stopped) return 'Stop is latched. Reset before continuing.';
    return enabled ? 'Motors enabled. Holding between actions.' : 'Motors are disabled. No motion is active.';
  }

  function motionButton() {
    if (stopped) {
      return (
        <button
          className="button wide"
          onClick={() => command('reset_stop')}
          disabled={!mine || !connected || busy || (!ready && enabled)}
        >
          <RotateCcw size={16} />
          Reset stop
        </button>
      );
    }
    if (!enabled) {
      return (
        <button
          className="button primary wide"
          onClick={() => command('enable')}
          disabled={!mine || busy || !connected || !camerasReady}
        >
          <Power size={16} />
          {state?.phase === 'Enabling' ? 'Enabling…' : 'Enable Motors'}
        </button>
      );
    }
    return null;
  }

  return (
    <aside className="control-panel">
      <div className="panel-heading">
        <div>
          <span className="section-icon">
            <Settings2 size={17} />
          </span>
          <h2>Control station</h2>
        </div>
        <span className="online-dot" />
      </div>
      <div className="control-content">
        <div className={`status-card ${stopped ? 'is-stopped' : ''}`}>
          <span className="status-label">ROBOT STATUS</span>
          <div>
            <span className={`status-light ${busy ? 'pulsing' : ''}`} />
            <strong>{connected ? state?.phase || 'Connecting' : 'Unknown'}</strong>
          </div>
          <p>{statusMessage()}</p>
        </div>
        {!mine && connected && (
          <div className="readonly">
            <Eye size={14} />
            Read-only tab
            <button onClick={onReconnect}>Take control</button>
          </div>
        )}
        {fault && (
          <button
            className="button primary wide"
            onClick={() => command('recover_control')}
            disabled={!connected || !mine || busy || !!pending}
          >
            {recovering ? <LoaderCircle className="spin" size={16} /> : <RotateCcw size={16} />}{' '}
            {recovering ? 'Recovering…' : 'Recover control'}
          </button>
        )}
        {motionButton()}
        <button className="button wide" onClick={() => onHome(true)} disabled={homeDisabled}>
          <Home size={15} />
          Home
        </button>
        {keepGrippers && (
          <button
            className="button wide"
            onClick={() => onHome(false)}
            disabled={homeDisabled}
            title="Return the arms to Home without opening or closing the grippers"
          >
            <Home size={15} />
            Home (keep grippers)
          </button>
        )}
        <button className="button subtle wide" onClick={onDisable} disabled={!mine || !enabled || !!pending}>
          <Power size={14} />
          Disable Motors
        </button>
        <button
          className="button subtle wide"
          onClick={() => command('reconnect_cameras')}
          disabled={!connected || !mine || busy || !!pending}
          title="Available while paused or idle; motors keep holding."
        >
          <RotateCcw size={14} />
          {state?.phase === 'Reconnecting cameras' ? 'Reconnecting cameras…' : 'Reconnect cameras'}
        </button>
        <div className="control-settings">
          <label className="field-label">
            Model
            <select
              aria-label="Model"
              value={state?.model ?? ''}
              disabled={!mine || busy || !!task}
              onChange={(e) => command('model', {model: e.target.value})}
            >
              {state?.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <PerformancePanel
            data={state?.performance}
            context={context}
            connected={connected}
            active={INFERENCE_PHASES.includes(state?.phase || '')}
            started={state?.inference_started_at}
          />
          <div className="model-health">
            <span>
              {state?.model_health.status === 'ok'
                ? 'Service ready'
                : state?.model_health.status === 'offline'
                  ? 'Service offline · retrying automatically'
                  : 'Connection not checked'}
            </span>
            <button onClick={() => Api.get('/api/model/health').catch((e) => onError(e.message))}>
              Check connection
            </button>
          </div>
          <div className="control-divider" />
        </div>
      </div>
    </aside>
  );
}
