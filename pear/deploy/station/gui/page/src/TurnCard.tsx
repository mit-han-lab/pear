import {Fragment, useEffect, useState} from 'react';
import {Check, ChevronRight, X} from 'lucide-react';
import {Turns, type Turn} from './turns';
import './TurnCard.css';

export const ENDED_PHASES = ['Failed', 'Interrupted', 'Context limit']; // a turn in one of these never continues
const CLOCK_TICK_MS = 250;
const SECONDS_DIGITS = 1;
const OUT_OF_REACH = 'UnreachablePoseError'; // the error of an action IK rejected before any motion
const seconds = (value: number) => `${value.toFixed(SECONDS_DIGITS)} s`;
const tokenCount = (value: number) => `${value.toLocaleString('en-US')} tokens`;

type TurnCardProps = {
  turn: Turn;
  active: boolean;
  phase?: string;
  format?: string;
  connected?: boolean;
};

/** One model turn: its thought, answer and action and what the action did; what the model saw folds away. */
export function TurnCard({turn: t, active, phase, format, connected = true}: TurnCardProps) {
  const [thoughtOpen, setThoughtOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const {text, block} = Turns.split(t.answer);
  const thought = t.answered && t.inferenceStarted ? t.answered - t.inferenceStarted : null;
  const generated = Turns.generated(t);
  const currentPhase = phase || t.phase || 'Observing';
  return (
    <article className={`turn-card ${active ? 'turn-active' : ''}`} data-turn-number={t.number}>
      <div className="turn-head">
        <span className="turn-badge">{t.number ?? '—'}</span>
        <strong>Turn {t.number ?? '—'}</strong>
        {(active || !!t.started) && (
          <span>· {active ? 'now' : new Date(t.started! * 1000).toLocaleTimeString('en-US', {hour12: false})}</span>
        )}
        {t.observation && (
          <button
            className="disclosure details-toggle"
            aria-expanded={detailsOpen}
            onClick={() => setDetailsOpen(!detailsOpen)}
          >
            Observation & State
            <ChevronRight size={13} />
          </button>
        )}
      </div>
      {t.observation && detailsOpen && (
        <div className="turn-details">
          <div className="turn-frames">
            {Object.entries(t.observation.images || {}).map(([role, url]) => (
              <a key={role} href={url} target="_blank" rel="noreferrer">
                <img loading="lazy" src={url} alt={`${role} turn ${t.number}`} />
                <span>{role.replaceAll('_', ' ')}</span>
              </a>
            ))}
          </div>
          <pre className="turn-json">{JSON.stringify(t.observation.state, null, 2)}</pre>
        </div>
      )}
      {t.reasoning && (
        <button
          className="disclosure thought-toggle"
          aria-expanded={thoughtOpen}
          onClick={() => setThoughtOpen(!thoughtOpen)}
        >
          <ChevronRight size={14} />
          {thought === null ? 'Reasoning' : `Thought for ${seconds(thought)}`}
          {typeof generated.count === 'number' &&
            ` · ${generated.isLowerBound ? '≥ ' : ''}${tokenCount(generated.count)}`}
        </button>
      )}
      {thoughtOpen && <div className="thought">{t.reasoning}</div>}
      {text && <p className="turn-text">{text}</p>}
      {block && <ActionCard block={block} format={format} />}
      {active ? (
        <TurnProgress
          phase={currentPhase}
          since={t.phaseStarted ?? t.started}
          tokens={t.performance?.output_tokens}
          connected={connected}
        />
      ) : (
        <TurnOutcome turn={t} />
      )}
    </article>
  );
}

/** The action's json, one part per line, with every non-zero quantity marked. */
function ActionCard({block, format}: {block: string; format?: string}) {
  let parts: [string, unknown][] = [];
  try {
    const value = JSON.parse(block);
    if (value && typeof value === 'object' && !Array.isArray(value)) parts = Object.entries(value);
  } catch {}
  return (
    <div className="action-card">
      <div className="action-head">
        <span>Action</span>
        {format && <code>{format}</code>}
      </div>
      <pre>
        {parts.length
          ? parts.map(([name, value], i) => (
              <Fragment key={name}>
                {i ? ' ' : '{'}
                {JSON.stringify(name)}:{' '}
                <ActionPart value={value} />
                {i < parts.length - 1 ? ',\n' : '}'}
              </Fragment>
            ))
          : block}
      </pre>
    </div>
  );
}

function ActionPart({value}: {value: unknown}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return <ActionValue value={value} />;
  return (
    <>
      {'{'}
      {Object.entries(value).map(([name, quantity], i) => (
        <Fragment key={name}>
          {i ? ', ' : ''}
          {JSON.stringify(name)}: <ActionValue value={quantity} />
        </Fragment>
      ))}
      {'}'}
    </>
  );
}

function ActionValue({value}: {value: unknown}) {
  const text = JSON.stringify(value);
  return typeof value === 'number' && value !== 0 ? <mark>{text}</mark> : <>{text}</>;
}

/** The running turn's phase, the time it has spent in it, and the tokens generated so far. */
function TurnProgress({
  phase,
  since,
  tokens,
  connected,
}: {
  phase: string;
  since?: number;
  tokens?: number | null;
  connected: boolean;
}) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!connected) return;
    const timer = setInterval(() => setNow(Date.now() / 1000), CLOCK_TICK_MS);
    return () => clearInterval(timer);
  }, [connected]);
  const parts = [phase];
  if (since) parts.push(seconds(Math.max(0, now - since)));
  if (typeof tokens === 'number') parts.push(tokenCount(tokens));
  return (
    <div className="turn-progress">
      <span className="turn-dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      {parts.join(' · ')}
    </div>
  );
}

/** How a finished turn ended: completed, with its arrival and grippers, or rejected, failed or interrupted, and why. */
function TurnOutcome({turn: t}: {turn: Turn}) {
  const result = t.result;
  if (!result && !ENDED_PHASES.includes(t.phase || '')) return null;

  const status = t.phase === 'Interrupted' ? 'interrupted' : result?.status || 'failed';
  const isOutOfReach = result?.error?.code === OUT_OF_REACH;
  const reason = isOutOfReach ? `Nothing moved. ${result?.error?.message}` : result?.error?.message || t.notice || '';
  const labels: Record<string, string> = {
    completed: 'Completed',
    rejected: isOutOfReach ? 'Rejected · out of reach' : 'Rejected',
    interrupted: 'Interrupted',
    failed: t.phase === 'Context limit' ? 'Context limit' : 'Failed',
  };
  const grippers = result?.gripper_outcomes
    ? Object.entries(result.gripper_outcomes)
    : result?.gripper_outcome
      ? [['', result.gripper_outcome]]
      : [];
  const duration = t.executed && t.executionStarted ? t.executed - t.executionStarted : null;
  return (
    <>
      <div className="turn-outcome">
        <span className={`result-chip ${status}`}>
          {status === 'completed' ? (
            <Check size={12} strokeWidth={2.6} />
          ) : status === 'rejected' ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6">
              <path d="M12 6v8M12 18v.5" strokeLinecap="round" />
            </svg>
          ) : (
            <X size={12} strokeWidth={2.6} />
          )}
          {labels[status] ?? status}
        </span>
        {status === 'completed' && (
          <>
            <span className="outcome-chip">{result?.settled ? 'Settled' : 'Not settled'}</span>
            <span className="outcome-chip">{result?.reached ? 'Reached' : 'Not reached'}</span>
            {grippers.map(([side, outcome]) => (
              <span className="outcome-chip" key={side}>
                {side ? `${side[0].toUpperCase()}${side.slice(1)} gripper` : 'Gripper'}:{' '}
                {outcome.replaceAll('_', ' ')}
              </span>
            ))}
            {duration !== null && <span className="outcome-note duration">{seconds(duration)}</span>}
          </>
        )}
        {reason && <span className="outcome-note">{reason}</span>}
      </div>
      {t.notice && t.notice !== reason && <p className="turn-note">{t.notice}</p>}
    </>
  );
}
