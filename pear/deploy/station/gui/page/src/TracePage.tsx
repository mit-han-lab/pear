import {useState} from 'react';
import {ChevronsDownUp, ChevronsUpDown} from 'lucide-react';
import {Turns} from './turns';
import {TurnCard} from './TurnCard';
import type {StreamEvent, Task} from './protocol';
import './theme.css';
import './Conversation.css';
import './TracePage.css';

/** What the page's service puts into a task's trace page. */
export type TraceData = {
  robot: string;
  task: Task & {reasoning_effort?: string};
  system_prompt: string | null;
  events: StreamEvent[];
  /** The conversation's tokens after its last answered turn, as the styled videos show them. */
  token_usage: number | null;
};

const OUTCOMES: [string, string][] = [
  ['completed', 'Completed'],
  ['rejected', 'Rejected'],
  ['failed', 'Failed'],
  ['interrupted', 'Interrupted'],
];
const SECONDS_PER_MINUTE = 60;

/** A task's trace to share: its record, then the conversation with every turn the model answered. */
export function TracePage({trace}: {trace: TraceData}) {
  const [isOpen, setOpen] = useState(false);
  const [foldVersion, setFoldVersion] = useState(0); // a new version folds or unfolds every part at once
  const {task, events} = trace;
  const turns = Turns.fromEvents(events).filter((t) => t.answered);
  const notices = events.filter((e) => Turns.NOTICE_TYPES.includes(e.type) && !e.data.attempt);
  const counts = new Map<string, number>();
  for (const t of turns) {
    const outcome = t.phase === 'Interrupted' ? 'interrupted' : t.result?.status;
    if (outcome) counts.set(outcome, (counts.get(outcome) ?? 0) + 1);
  }
  // the task ran until its last turn ended; a task cleared later stood still until then
  const lastTurnTime = events.filter((e) => e.data.attempt).pop()?.time;
  const seconds = lastTurnTime ? Math.max(0, Math.round(lastTurnTime - task.created)) : null;
  const meta = [
    ['Model', task.model_label || task.model],
    ['Reasoning effort', task.reasoning_effort],
    ['Started', new Date(task.created * 1000).toLocaleString('en-US', {hour12: false, timeZoneName: 'short'})],
    [
      'Duration',
      seconds === null ? '' : `${Math.floor(seconds / SECONDS_PER_MINUTE)} min ${seconds % SECONDS_PER_MINUTE} s`,
    ],
    ...OUTCOMES.map(([outcome, label]) => [label, counts.get(outcome)?.toString()]),
  ].filter(([, value]) => value);

  function foldAll(open: boolean) {
    setOpen(open);
    setFoldVersion((version) => version + 1);
  }

  return (
    <main className="trace-page">
      <header className="trace-header">
        <span className="trace-eyebrow">{trace.robot} episode trace</span>
        <h1>{task.instruction}</h1>
        <dl className="trace-meta">
          {meta.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <p className="trace-summary">
          <b>{turns.length}</b> turns
          {trace.token_usage !== null && (
            <>
              {' · '}
              <b>{trace.token_usage.toLocaleString('en-US')}</b> token usage
            </>
          )}
        </p>
        <div className="trace-toolbar">
          <button onClick={() => foldAll(true)}>
            <ChevronsUpDown size={14} /> Expand all
          </button>
          <button onClick={() => foldAll(false)}>
            <ChevronsDownUp size={14} /> Collapse all
          </button>
        </div>
      </header>
      <div className="chat-thread">
        {trace.system_prompt && (
          <details className="trace-system" key={`system-${foldVersion}`} open={isOpen}>
            <summary>System prompt</summary>
            <pre>{trace.system_prompt}</pre>
          </details>
        )}
        <div className="user-turn">
          <div className="user-bubble">{task.instruction}</div>
          {!!task.images?.length && (
            <div className="trace-images">
              {task.images.map((image) => (
                <img key={image.id} src={image.url} alt={image.name} />
              ))}
            </div>
          )}
        </div>
        {notices.map((e) => (
          <p className="chat-notice" key={e.id}>
            {e.data.message}
            {e.data.reason ? ` · ${e.data.reason}` : ''}
          </p>
        ))}
        {turns.map((t) => (
          <TurnCard
            key={`${t.id}-${foldVersion}`}
            turn={t}
            format={task.action_format}
            active={false}
            isOpen={isOpen}
          />
        ))}
      </div>
    </main>
  );
}
