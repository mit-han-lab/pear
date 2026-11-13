import {useEffect, useRef, useState} from 'react';
import {Activity, ArrowRight, Bot, Eye, Plus} from 'lucide-react';
import {Api} from './api';
import {HistoryFilter, TaskAnnotations} from './Tags';
import {HistoryFooter, HistoryList, type History} from './History';
import {Recordings} from './Recordings';
import {SavedImages} from './TaskImages';
import {ENDED_PHASES, TurnCard} from './TurnCard';
import type {Annotation, ServerState, StreamEvent, TagCount, Task} from './protocol';
import type {Turn} from './turns';
import './Conversation.css';

const FOLLOW_DISTANCE_PX = 90; // closer than this to the bottom keeps the view following new output
const MAX_TURNS_SHOWN = 200;
const MAX_NOTICES_SHOWN = 8;
const NOTICE_TYPES = ['session', 'notice', 'interrupted'];

type ConversationProps = {
  isHistory: boolean;
  state: ServerState | null;
  connected: boolean;
  mine: boolean;
  pending: string;
  activeTask: Task | null | undefined;
  activeEvents: StreamEvent[];
  turns: Turn[];
  history: History;
  tagCounts: TagCount[];
  follow: boolean;
  events: StreamEvent[];
  onFollow: (follow: boolean) => void;
  onClear: () => void;
  onPreview: (url: string) => void;
};

/** The task conversation, or the list of saved tasks, with the button that starts a new task. */
export function Conversation(props: ConversationProps) {
  const {isHistory, state, connected, activeTask, activeEvents, turns, history, tagCounts} = props;
  const scroll = useRef<HTMLDivElement>(null);
  const [savedTags, setSavedTags] = useState<{taskId: string; tags: string[]} | null>(null);
  const task = state?.task;
  const busy = !!state?.busy;
  const lastTurn = turns[turns.length - 1];
  const activeTurn =
    !isHistory && busy && lastTurn && !lastTurn.result && !ENDED_PHASES.includes(lastTurn.phase || '')
      ? lastTurn
      : undefined;
  const liveTaskId = isHistory ? undefined : activeTask?.id;
  const notices = activeEvents.filter((e) => NOTICE_TYPES.includes(e.type) && !e.data.attempt);
  const progress = [lastTurn?.number && `Turn ${lastTurn.number}`, busy ? 'Running' : state?.phase]
    .filter(Boolean)
    .join(' · ');
  useEffect(() => {
    if (props.follow && scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [props.events, props.follow]);
  // Tag counts refresh after every tag edit, so the running task's tags reload with them.
  useEffect(() => {
    if (!liveTaskId) return;
    let stopped = false;
    Api.get<Annotation>(`/api/history/${liveTaskId}/annotations`)
      .then((value) => {
        if (!stopped) setSavedTags({taskId: liveTaskId, tags: value.tags});
      })
      .catch(() => {});
    return () => {
      stopped = true;
    };
  }, [liveTaskId, tagCounts]);

  function trackScroll() {
    const node = scroll.current;
    if (node) props.onFollow(node.scrollHeight - node.scrollTop - node.clientHeight < FOLLOW_DISTANCE_PX);
  }

  return (
    <>
      <div className="conversation-heading">
        <div>
          <h2>{isHistory && !history.opened ? 'Task history' : activeTask?.instruction || 'Task conversation'}</h2>
          {!isHistory && task && (
            <span className={`turn-pill ${busy ? 'running' : ''}`}>
              <i />
              {progress}
            </span>
          )}
        </div>
        {!isHistory && (
          <button className="new-task" onClick={props.onClear} disabled={!props.mine || !task || !!props.pending}>
            <Plus size={15} strokeWidth={2.2} />
            New task
          </button>
        )}
      </div>
      {isHistory && !history.opened && (
        <HistoryFilter
          counts={history.counts}
          selected={history.tagFilter}
          onChange={history.setTagFilter}
          shown={history.shown.length}
          total={history.rows.length}
        />
      )}
      <div className="conversation-scroll" ref={scroll} onScroll={trackScroll}>
        {isHistory && !history.opened ? (
          <HistoryList history={history} />
        ) : !activeTask ? (
          <EmptyState />
        ) : (
          <div className="chat-thread">
            <div className="user-turn">
              <div className="user-bubble">{activeTask.instruction}</div>
              <SavedImages images={activeTask.images} onPreview={props.onPreview} />
              {savedTags?.taskId === liveTaskId && !!savedTags?.tags.length && (
                <div className="user-tags">
                  {savedTags.tags.map((tag) => (
                    <span key={tag}>{tag}</span>
                  ))}
                </div>
              )}
            </div>
            {isHistory && (
              <TaskAnnotations taskId={activeTask.id} suggestions={tagCounts} onSaved={history.saveAnnotations} />
            )}
            {notices.slice(-MAX_NOTICES_SHOWN).map((e) => (
              <p className="chat-notice" key={e.id}>
                {e.data.message}
                {e.data.reason ? ` · ${e.data.reason}` : ''}
              </p>
            ))}
            {turns.slice(-MAX_TURNS_SHOWN).map((t) => (
              <TurnCard
                key={t.id}
                turn={t}
                format={activeTask.action_format ?? state?.action_format}
                connected={isHistory || connected}
                active={t === activeTurn}
                phase={state?.active_turn?.attempt === t.id && state?.phase === 'Settling' ? 'Settling' : t.phase}
              />
            ))}
            {busy && !isHistory && !activeTurn && (
              <div className="turn-progress">
                <span className="turn-dots" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
                {state?.phase}…
              </div>
            )}
            <Recordings taskId={activeTask.id} />
            {isHistory && history.opened && <HistoryFooter history={history} task={history.opened} />}
          </div>
        )}
      </div>
    </>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-orbit">
        <Bot size={30} />
        <span />
        <i />
      </div>
      <h3>What should we work on?</h3>
      <p>
        Describe a task. Watch the model observe, reason,
        <br />
        and act — one measured step at a time.
      </p>
      <div className="empty-steps">
        <span>
          <Eye size={14} />
          Observe
        </span>
        <ArrowRight size={12} />
        <span>
          <Bot size={14} />
          Reason
        </span>
        <ArrowRight size={12} />
        <span>
          <Activity size={14} />
          Act
        </span>
      </div>
    </div>
  );
}
