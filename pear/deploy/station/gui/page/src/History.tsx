import {useState} from 'react';
import {ChevronRight, Clock3, Download} from 'lucide-react';
import {Api} from './api';
import {countTags, hasTag, TagChip} from './Tags';
import type {Annotation, StreamEvent, Task} from './protocol';
import './History.css';

const PAGE_SIZE = 100; // events the service returns per page of a saved task

/** The saved tasks, the one opened and its stored events. */
export function useHistory(refreshTags: () => void, showError: (message: string) => void) {
  const [rows, setRows] = useState<Task[]>([]);
  const [opened, setOpened] = useState<Task | null>(null);
  const [pastEvents, setPastEvents] = useState<StreamEvent[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [tagFilter, setTagFilter] = useState<string[]>([]);

  async function load() {
    setOpened(null);
    try {
      setRows(await Api.get<Task[]>('/api/history'));
    } catch (e) {
      showError((e as Error).message);
    }
  }

  /** The first page of a task's events, or all of them when `everything` is set. */
  async function open(task: Task, everything = false) {
    setOpened(task);
    const events = await Api.get<StreamEvent[]>(`/api/history/${task.id}/events${everything ? '?all=1' : ''}`);
    setPastEvents(events);
    setHasMore(!everything && events.length === PAGE_SIZE);
  }

  function saveAnnotations(id: string, value: Annotation) {
    setRows((old) => old.map((row) => (row.id === id ? {...row, ...value} : row)));
    setOpened((task) => (task && task.id === id ? {...task, ...value} : task));
    refreshTags();
  }

  const shown = rows.filter((row) => tagFilter.every((tag) => hasTag(row.tags, tag)));
  return {
    rows,
    shown,
    counts: countTags(rows),
    opened,
    pastEvents,
    hasMore,
    tagFilter,
    setTagFilter,
    load,
    open,
    saveAnnotations,
  };
}

export type History = ReturnType<typeof useHistory>;

export function HistoryList({history}: {history: History}) {
  if (history.rows.length === 0) return <div className="empty-small">No saved tasks yet.</div>;
  if (history.shown.length === 0) return <div className="empty-small">No tasks have all selected tags.</div>;
  return (
    <div className="history-list">
      {history.shown.map((task) => (
        <button key={task.id} onClick={() => history.open(task)}>
          <Clock3 size={18} />
          <div>
            <strong>{task.instruction}</strong>
            <span>
              {new Date(task.created * 1000).toLocaleString('en-US')} · {task.model_label || task.model} ·{' '}
              {task.backend === 'fake' ? 'Simulation' : 'Live Robot'}
            </span>
            {!!task.tags?.length && (
              <div className="history-tags">
                {task.tags.map((tag) => (
                  <TagChip key={tag} tag={tag} />
                ))}
              </div>
            )}
          </div>
          <ChevronRight size={17} />
        </button>
      ))}
    </div>
  );
}

export function HistoryFooter({history, task}: {history: History; task: Task}) {
  return (
    <div className="history-footer">
      {history.hasMore && <button onClick={() => history.open(task, true)}>Show all records</button>}
      <a href={`/api/history/${task.id}/export`}>
        <Download size={15} />
        Export task
      </a>
    </div>
  );
}
