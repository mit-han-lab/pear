import {useEffect, useRef, useState, type ClipboardEvent, type DragEvent, type FocusEvent} from 'react';
import {ArrowDown, ArrowUp, Play, Square, Tag} from 'lucide-react';
import {TagEditor, TaskAnnotations} from './Tags';
import {ImageAttachments, useTaskImages} from './TaskImages';
import type {Annotation, ServerState, TagCount} from './protocol';
import './Composer.css';

const MAX_INSTRUCTION_CHARS = 20000;
// The one round button follows the task: start it, pause it while it runs, resume it once paused.
const TASK_ACTIONS = {
  send: {label: 'Send and start', icon: <ArrowUp size={18} />},
  pause: {label: 'Pause task', icon: <Square size={13} fill="currentColor" />},
  resume: {label: 'Resume task', icon: <Play size={15} fill="currentColor" />},
};

/** The unsent task: its text, tags and images, and the id that makes a resent start idempotent. */
export function useDraft(clientId: string) {
  const [text, setText] = useState('');
  // Tags stay for the next task: consecutive evaluations usually share a setup.
  const [tags, setTags] = useState<string[]>([]);
  const images = useTaskImages(clientId);
  const startId = useRef<string | null>(null);
  useEffect(() => {
    startId.current = null;
  }, [text, images.images, tags]);

  function clear() {
    setText('');
    images.reset();
  }

  return {text, setText, tags, setTags, images, startId, clear};
}

type Draft = ReturnType<typeof useDraft>;

type ComposerProps = {
  state: ServerState | null;
  connected: boolean;
  mine: boolean;
  pending: string;
  enabled: boolean;
  canRun: boolean;
  draft: Draft;
  tagCounts: TagCount[];
  follow: boolean;
  onFollow: (follow: boolean) => void;
  command: (name: string, data?: Record<string, unknown>) => Promise<boolean>;
  onError: (message: string) => void;
  onTagsChanged: () => void;
  onAnnotationsSaved: (taskId: string, value: Annotation) => void;
  onPreview: (url: string) => void;
};

/** The task box: instruction, images, tags and recording, and the one button that starts, pauses or resumes. */
export function Composer(props: ComposerProps) {
  const {state, connected, mine, pending, enabled, draft, tagCounts} = props;
  const [recordingChoice, setRecordingChoice] = useState<boolean | null>(null);
  const [tagsOpen, setTagsOpen] = useState(false);
  const tagsPanel = useRef<HTMLDivElement>(null);
  const tagsButton = useRef<HTMLButtonElement>(null);
  const task = state?.task;
  const busy = !!state?.busy;
  const referenceImages = state?.supports_reference_images === true;
  const recordingOn = recordingChoice ?? !!state?.recording.enabled;
  const canAttach = connected && mine && !task && !pending && referenceImages;
  const canSend =
    props.canRun &&
    !task &&
    !!draft.text.trim() &&
    !pending &&
    draft.images.ready &&
    (!draft.images.images.length || referenceImages);
  const taskAction = busy ? 'pause' : task ? 'resume' : 'send';
  const canControlTask = mine && enabled && state?.robot.status !== 'FAULT' && !state?.stop_latched && !pending;
  useEffect(() => {
    if (recordingChoice !== null && state?.recording.enabled === recordingChoice) setRecordingChoice(null);
  }, [state?.recording.enabled, recordingChoice]);
  useEffect(() => {
    if (tagsOpen) tagsPanel.current?.querySelector('input')?.focus();
  }, [tagsOpen]);

  function addImages(files: File[]) {
    if (!canAttach) return;
    const message = draft.images.add(files);
    if (message) props.onError(message);
  }

  async function send() {
    if (!canSend) return;
    if (draft.images.expired()) {
      draft.images.invalidate();
      props.onError('Task images expired; retry the uploads before sending.');
      return;
    }
    const started = await props.command('start', {
      instruction: draft.text,
      image_ids: draft.images.ids,
      tags: draft.tags,
    });
    if (started) {
      draft.images.reset();
      draft.startId.current = null;
      props.onTagsChanged();
    }
  }

  function toggleRecording() {
    const recordingChosen = !recordingOn;
    setRecordingChoice(recordingChosen);
    props.command('recording', {enabled: recordingChosen}).then((ok) => {
      if (!ok) setRecordingChoice(null);
    });
  }

  function statusPhrase() {
    if (busy) return task ? 'Running' : state?.phase;
    if (task) return 'Paused · New task to start another';
    return enabled ? 'Continuous execution' : 'Enable motors to begin';
  }

  // The tag panel closes once focus leaves it; the tag editor adds the typed text on that same blur.
  function leaveTags(e: FocusEvent) {
    if (!e.currentTarget.contains(e.relatedTarget)) setTagsOpen(false);
  }

  function dragOver(e: DragEvent) {
    if (!Array.from(e.dataTransfer.types).includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = canAttach ? 'copy' : 'none';
  }

  function paste(e: ClipboardEvent) {
    const files = Array.from(e.clipboardData.items)
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null);
    if (!files.length) return;
    e.preventDefault();
    addImages(files);
  }

  return (
    <div className="composer-area">
      {!props.follow && (
        <button className="jump" onClick={() => props.onFollow(true)}>
          <ArrowDown size={13} />
          Latest
        </button>
      )}
      <div
        className="composer"
        onDragOver={dragOver}
        onDrop={(e) => {
          e.preventDefault();
          addImages(Array.from(e.dataTransfer.files));
        }}
      >
        <textarea
          aria-label="Task instruction"
          placeholder="Describe the task you want the robot to perform…"
          value={draft.text}
          onChange={(e) => draft.setText(e.target.value)}
          maxLength={MAX_INSTRUCTION_CHARS}
          onPaste={paste}
          onKeyDown={(e) => {
            if (e.key !== 'Enter' || e.shiftKey || e.nativeEvent.isComposing) return;
            e.preventDefault();
            void send();
          }}
        />
        {!task && connected && !referenceImages && (
          <div className="image-support-note">Task images require an updated model service.</div>
        )}
        <div className="composer-tools">
          <ImageAttachments draft={draft.images} disabled={!canAttach} onFiles={addImages} onPreview={props.onPreview} />
          <div
            className="composer-tags"
            onBlur={leaveTags}
            onKeyDown={(e) => {
              if (e.key !== 'Escape' || !tagsOpen) return;
              setTagsOpen(false);
              tagsButton.current?.focus();
            }}
          >
            <button
              ref={tagsButton}
              type="button"
              className="composer-pill"
              aria-expanded={tagsOpen}
              aria-haspopup="dialog"
              onClick={() => setTagsOpen(!tagsOpen)}
            >
              <Tag size={14} aria-hidden />
              Tags · {draft.tags.length}
            </button>
            {tagsOpen && (
              <div className="tags-popover" role="dialog" aria-label="Task tags" tabIndex={-1} ref={tagsPanel}>
                <TagEditor
                  tags={draft.tags}
                  suggestions={tagCounts}
                  onChange={draft.setTags}
                  label="Tags for the new task"
                  scope="Next task"
                  menuAbove
                />
                {task && (
                  <TaskAnnotations compact taskId={task.id} suggestions={tagCounts} onSaved={props.onAnnotationsSaved} />
                )}
              </div>
            )}
          </div>
          {state && (
            <button
              type="button"
              role="switch"
              aria-checked={recordingOn}
              className={`composer-pill recording-pill ${recordingOn ? 'on' : ''}`}
              disabled={!connected || !mine || busy || !!pending}
              title="Record a full video and a motion-only video of each run"
              onClick={toggleRecording}
            >
              <i className={state.recording.active?.status === 'recording' ? 'pulsing' : ''} />
              Recording
            </button>
          )}
          <div className="composer-submit">
            <span>{statusPhrase()}</span>
            <button
              className="task-action"
              aria-label={TASK_ACTIONS[taskAction].label}
              title={TASK_ACTIONS[taskAction].label}
              disabled={taskAction === 'send' ? !canSend : !canControlTask}
              onClick={() => (taskAction === 'send' ? void send() : void props.command(taskAction))}
            >
              {TASK_ACTIONS[taskAction].icon}
            </button>
          </div>
        </div>
      </div>
      <div className="composer-hint">Enter to send · Shift + Enter for a new line</div>
    </div>
  );
}
