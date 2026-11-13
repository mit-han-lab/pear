import React, {useEffect, useId, useRef, useState} from 'react';
import {ChevronDown, GripVertical, Tag, X} from 'lucide-react';
import {Api} from './api';
import type {Annotation, TagCount} from './protocol';
import './Tags.css';

const MAX_TAGS = 20,
  MAX_TAG = 40;
const same = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();
export const hasTag = (tags: string[] | undefined, tag: string) => !!tags?.some((t) => same(t, tag));

/** The same tag always gets the same colour, in every list and task. */
function hue(tag: string) {
  let value = 0;
  for (const c of tag.toLowerCase()) value = (value * 31 + c.codePointAt(0)!) >>> 0;
  return value % 360;
}

export function TagChip({
  tag,
  count,
  selected,
  onClick,
  onRemove,
  grip,
  ...attributes
}: {
  tag: string;
  count?: number;
  selected?: boolean;
  onClick?: () => void;
  onRemove?: () => void;
  grip?: boolean;
} & React.HTMLAttributes<HTMLSpanElement>) {
  const style = {'--tag-hue': hue(tag)} as React.CSSProperties;
  if (onClick)
    return (
      <button
        type="button"
        className={`tag-chip ${selected ? 'selected' : ''}`}
        style={style}
        aria-pressed={selected}
        onClick={onClick}
      >
        {tag}
        {count != null && <small>{count}</small>}
      </button>
    );
  return (
    <span className="tag-chip" style={style} {...attributes}>
      {grip && <GripVertical size={11} aria-hidden />}
      {tag}
      {onRemove && (
        <button type="button" aria-label={`Remove tag ${tag}`} onClick={onRemove}>
          <X size={11} />
        </button>
      )}
    </span>
  );
}

/** Free-text tags with a menu of earlier tags, most used first; a case variant of an
 *  existing tag reuses its spelling. Enter adds the typed text unless an earlier tag
 *  was chosen with the arrow keys, so typing never gets silently replaced. */
export function TagEditor({
  tags,
  suggestions,
  onChange,
  disabled,
  label,
  scope,
  menuAbove,
}: {
  tags: string[];
  suggestions: TagCount[];
  onChange: (tags: string[]) => void;
  disabled?: boolean;
  label: string;
  scope?: string;
  menuAbove?: boolean;
}) {
  const [text, setText] = useState(''),
    [open, setOpen] = useState(false),
    [active, setActive] = useState(-1),
    menu = useId();
  const input = useRef<HTMLInputElement>(null),
    dragged = useRef(-1);
  const query = text.split(/\s+/).filter(Boolean).join(' ');
  const options = suggestions.filter((s) => !hasTag(tags, s.tag) && s.tag.toLowerCase().includes(query.toLowerCase()));
  const known = !!query && (hasTag(tags, query) || suggestions.some((s) => same(s.tag, query)));
  const shown = open && (options.length > 0 || (!!query && !known));
  function choose(tag: string) {
    setText('');
    setActive(-1);
    if (!tag || hasTag(tags, tag) || tags.length >= MAX_TAGS) return;
    onChange([...tags, suggestions.find((s) => same(s.tag, tag))?.tag ?? tag]);
  }
  function move(from: number, to: number) {
    if (from === to || to < 0 || to >= tags.length) return;
    const next = [...tags];
    next.splice(to, 0, next.splice(from, 1)[0]);
    onChange(next);
  }
  function tagKey(e: React.KeyboardEvent<HTMLSpanElement>, index: number) {
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      onChange(tags.filter((_, i) => i !== index));
      input.current?.focus();
    } else if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      e.preventDefault();
      move(index, index + (e.key === 'ArrowLeft' ? -1 : 1));
    }
  }
  function key(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Backspace' && !text && tags.length) {
      e.preventDefault();
      onChange(tags.slice(0, -1));
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      setActive((i) => (e.key === 'ArrowDown' ? Math.min(i + 1, options.length - 1) : Math.max(i - 1, -1)));
    } else if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      choose(shown && options[active] ? options[active].tag : query);
    } else if (e.key === 'Escape' && shown) {
      e.preventDefault();
      e.stopPropagation();
      setOpen(false);
      setActive(-1);
    }
  }
  return (
    <div className="tag-editor">
      <Tag size={13} aria-hidden />
      {scope && <span className="tags-scope">{scope}</span>}
      {tags.map((t, i) =>
        disabled ? (
          <TagChip key={t} tag={t} />
        ) : (
          <TagChip
            key={t}
            tag={t}
            grip
            tabIndex={0}
            draggable
            aria-label={`Tag ${t}`}
            onKeyDown={(e) => tagKey(e, i)}
            onDragStart={() => (dragged.current = i)}
            onDragOver={(e) => e.preventDefault()}
            onDrop={() => move(dragged.current, i)}
            onRemove={() => onChange(tags.filter((x) => x !== t))}
          />
        ),
      )}
      {!disabled && tags.length < MAX_TAGS && (
        <span className="tag-input">
          <input
            ref={input}
            aria-label={label}
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={shown}
            aria-controls={menu}
            aria-activedescendant={shown && options[active] ? `${menu}-${active}` : undefined}
            value={text}
            maxLength={MAX_TAG}
            placeholder={
              suggestions.length ? 'Type or pick a tag' : tags.length ? 'Add tag' : 'Add tags (Enter to add)'
            }
            onChange={(e) => {
              setText(e.target.value);
              setActive(-1);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onClick={() => setOpen(true)}
            onBlur={() => {
              setOpen(false);
              choose(query);
            }}
            onKeyDown={key}
          />
          {suggestions.length > 0 && (
            <button
              type="button"
              className="tag-menu-toggle"
              tabIndex={-1}
              aria-label="Show earlier tags"
              onMouseDown={(e) => {
                e.preventDefault();
                if (document.activeElement === input.current) setOpen((o) => !o);
                else input.current?.focus();
              }}
            >
              <ChevronDown size={13} />
            </button>
          )}
          {/* Mouse down would blur the input and add the typed text before the click. */}
          {shown && (
            <div className={`tag-menu ${menuAbove ? 'above' : ''}`} onMouseDown={(e) => e.preventDefault()}>
              <ul id={menu} role="listbox" aria-label="Earlier tags">
                {options.map((s, i) => (
                  <li
                    key={s.tag}
                    id={`${menu}-${i}`}
                    role="option"
                    aria-selected={i === active}
                    className={i === active ? 'active' : ''}
                    onMouseEnter={() => setActive(i)}
                    onClick={() => choose(s.tag)}
                  >
                    <TagChip tag={s.tag} />
                    <small>
                      {s.count} task{s.count === 1 ? '' : 's'}
                    </small>
                  </li>
                ))}
              </ul>
              {query && !known && (
                <div className="tag-menu-new" onClick={() => choose(query)}>
                  New tag “{query}” <kbd>Enter</kbd>
                </div>
              )}
            </div>
          )}
        </span>
      )}
      {disabled && !tags.length && <span className="tag-empty">No tags</span>}
    </div>
  );
}

/** Tags of one saved task; changes are written immediately and in order.
 *  Compact form sits in the composer, so a running task stays editable while the
 *  conversation auto-scrolls. */
export function TaskAnnotations({
  taskId,
  suggestions,
  onSaved,
  compact,
}: {
  taskId: string;
  suggestions: TagCount[];
  onSaved: (taskId: string, value: Annotation) => void;
  compact?: boolean;
}) {
  const [value, setValue] = useState<Annotation | null>(null),
    [error, setError] = useState(''),
    [unavailable, setUnavailable] = useState(false);
  const queue = useRef(Promise.resolve()),
    latest = useRef(0);
  useEffect(() => {
    let stopped = false;
    setValue(null);
    setError('');
    setUnavailable(false);
    Api.get<Annotation>(`/api/history/${taskId}/annotations`)
      .then((v) => {
        if (!stopped) setValue(v);
      })
      .catch(() => {
        if (!stopped) setUnavailable(true);
      });
    return () => {
      stopped = true;
    };
  }, [taskId]);
  function save(change: Annotation) {
    const request = ++latest.current,
      previous = value;
    if (value) setValue({...value, tags: change.tags});
    setError('');
    // Serialized so a quick second edit can never be overtaken by the first.
    const done = queue.current
      .then(async () => {
        const body = await Api.post<Annotation>(`/api/history/${taskId}/annotations`, change);
        onSaved(taskId, body);
        if (request === latest.current) setValue(body);
      })
      .catch((e) => {
        if (request === latest.current) {
          setError((e as Error).message);
          if (previous) setValue(previous);
        }
      });
    queue.current = done;
  }
  // A failed load is secondary information, not an alert competing with robot errors.
  const className = `task-annotations ${compact ? 'compact' : ''}`;
  if (!value)
    return unavailable ? (
      <div className={className}>
        <span className="tag-empty">Tags unavailable</span>
      </div>
    ) : null;
  return (
    <div className={className} aria-label="Task tags">
      <TagEditor
        tags={value.tags}
        suggestions={suggestions}
        label="Add a tag to this task"
        scope={compact ? 'This task' : undefined}
        menuAbove={compact}
        onChange={(tags) => save({tags})}
      />
      {error && (
        <p className="annotation-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

/** All-of filter: a task is shown only when it carries every selected tag. */
export function HistoryFilter({
  counts,
  selected,
  onChange,
  shown,
  total,
}: {
  counts: TagCount[];
  selected: string[];
  onChange: (tags: string[]) => void;
  shown: number;
  total: number;
}) {
  if (!counts.length) return null;
  return (
    <div className="history-filter" aria-label="Filter tasks by tag">
      <div className="history-filter-tags">
        {counts.map((c) => (
          <TagChip
            key={c.tag}
            tag={c.tag}
            count={c.count}
            selected={hasTag(selected, c.tag)}
            onClick={() =>
              onChange(hasTag(selected, c.tag) ? selected.filter((t) => !same(t, c.tag)) : [...selected, c.tag])
            }
          />
        ))}
      </div>
      <div className="history-filter-summary">
        {selected.length ? (
          <>
            {shown} of {total} tasks have all selected tags
            <button type="button" onClick={() => onChange([])}>
              Clear filter
            </button>
          </>
        ) : (
          <>{total} tasks · select tags to filter</>
        )}
      </div>
    </div>
  );
}

/** Same counting as the server: case variants count toward the oldest task's spelling. */
export function countTags(rows: {tags?: string[]; created?: number}[]): TagCount[] {
  const counts = new Map<string, TagCount>();
  const oldest = [...rows].sort((a, b) => (a.created || 0) - (b.created || 0));
  for (const row of oldest)
    for (const tag of row.tags || []) {
      const key = tag.toLowerCase(),
        entry = counts.get(key);
      if (entry) entry.count++;
      else counts.set(key, {tag, count: 1});
    }
  return [...counts.values()].sort(
    (a, b) => b.count - a.count || a.tag.toLowerCase().localeCompare(b.tag.toLowerCase()),
  );
}

/** Every tag of the saved tasks with its use count; refreshed once signed in and after each change. */
export function useTagCounts(authed: boolean) {
  const [tagCounts, setTagCounts] = useState<TagCount[]>([]);
  const refreshTags = () => {
    Api.get<TagCount[]>('/api/tags')
      .then(setTagCounts)
      .catch(() => {});
  };
  useEffect(() => {
    if (authed) refreshTags();
  }, [authed]);
  return {tagCounts, refreshTags};
}
