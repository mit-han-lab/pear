import {useEffect, useState} from 'react';
import type {PerformanceData} from './protocol';
import type {ContextUsage} from './turns';
import './PerformancePanel.css';

const CLOCK_TICK_MS = 250;
const NOT_PROVIDED = 'Not provided';
const number = (n: unknown, suffix: string) =>
  typeof n === 'number' && Number.isFinite(n) ? `${n.toFixed(2)} ${suffix}` : NOT_PROVIDED;
const count = (n: number) => n.toLocaleString('en-US');

/** The latest inference's timings and tokens, and how full the model's context is. */
export function PerformancePanel({
  data,
  context,
  active = false,
  connected = true,
  started,
}: {
  data?: PerformanceData | null;
  context: ContextUsage;
  active?: boolean;
  connected?: boolean;
  started?: number | null;
}) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!active || !connected) return;
    setNow(Date.now() / 1000);
    const timer = setInterval(() => setNow(Date.now() / 1000), CLOCK_TICK_MS);
    return () => clearInterval(timer);
  }, [active, connected, started]);
  const p = connected ? data : null;
  const lower = p?.final && p?.partial ? '≥ ' : '';
  // A context without a known limit shows only what it holds.
  const contextText =
    typeof context.used !== 'number'
      ? NOT_PROVIDED
      : typeof context.limit === 'number'
        ? `${count(context.used)} / ${count(context.limit)}`
        : count(context.used);
  let elapsed = p?.elapsed_s;
  if (active && connected && !p?.final) {
    if (typeof elapsed === 'number' && p?.received_at) elapsed += Math.max(0, now - p.received_at);
    else if (started) elapsed = Math.max(0, now - started);
  }
  return (
    <div className="performance" aria-label="Inference performance">
      {!connected && <p>Connection lost</p>}
      <div>
        <span>Client decode throughput</span>
        <strong>{number(p?.client_decode_throughput_tps, 'tok/s')}</strong>
      </div>
      <div>
        <span>Inference elapsed</span>
        <strong>{number(elapsed, 's')}</strong>
      </div>
      <div>
        <span>Generated tokens</span>
        <strong>
          {typeof p?.output_tokens === 'number'
            ? lower + p.output_tokens.toLocaleString('en-US')
            : active && connected
              ? 'Pending'
              : NOT_PROVIDED}
        </strong>
      </div>
      <div>
        <span>Context usage</span>
        <strong>{contextText}</strong>
      </div>
      {p?.final && p.partial && <p>Incomplete usage · confirmed counts are lower bounds</p>}
      {p?.final && p.status === 'interrupted' && <p>Interrupted · partial statistics</p>}
    </div>
  );
}
