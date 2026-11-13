import {useEffect, useRef, useState} from 'react';
import {Archive, Download, Film, Layers, LoaderCircle, Video, X} from 'lucide-react';
import {Api} from './api';
import type {MergedRecording, MergeState, RecordingSegment, Skip, VideoRender} from './protocol';
import './Recordings.css';

const REFRESH_MS = 1500;
const MERGED_PART = 'merged';
const RENDER_STAGES: Record<string, string> = {
  waiting: 'Waiting for the recording',
  queued: 'Queued',
  rendering: 'Generating',
};
const duration = (seconds?: number) => (seconds == null ? '' : `${seconds.toFixed(1)} s`);
const segmentCount = (count: number) => `${count} ${count === 1 ? 'segment' : 'segments'}`;
const segmentNames = (ids: string[]) => `${ids.length === 1 ? 'segment' : 'segments'} ${ids.map(Number).join(', ')}`;
const inProgress = (render?: VideoRender | null) => (render && render.status !== 'failed' ? render : null);
function skipNote(label: string, rows: Skip[]) {
  const byReason = new Map<string, string[]>();
  for (const row of rows) byReason.set(row.reason, [...(byReason.get(row.reason) ?? []), row.id]);
  return rows.length
    ? `${label} skips ${[...byReason].map(([reason, ids]) => `${segmentNames(ids)} (${reason})`).join('; ')}.`
    : '';
}

/** Where a render of styled videos is, with its progress. */
function RenderStage({render}: {render: VideoRender}) {
  const percent = render.progress == null ? '' : ` ${Math.round(render.progress * 100)}%`;
  return (
    <>
      <LoaderCircle className="spin" size={12} /> {RENDER_STAGES[render.status]}
      {percent}
    </>
  );
}

/** The play buttons and download links of a full and a motion-only video. */
function VideoPair({
  url,
  video,
  download,
  onPlay,
}: {
  url: string;
  video: MergedRecording;
  download: (label: string) => string;
  onPlay: (url: string) => void;
}) {
  return (
    <>
      {(
        [
          ['full', 'Full video', video.duration_s],
          ['motion', 'Motion only', video.motion_duration_s],
        ] as const
      ).map(([kind, label, seconds]) => {
        const name = video[kind];
        if (!name) return null;
        return (
          <div key={kind}>
            <button onClick={() => onPlay(`${url}/${name}?v=${video.created}`)}>
              {label} <span>{duration(seconds)}</span>
            </button>
            <a aria-label={download(label.toLowerCase())} href={`${url}/${name}?download=1`}>
              <Download size={15} />
            </a>
          </div>
        );
      })}
    </>
  );
}

/** The task's styled videos: each segment's once it is rendered, and all segments merged when the task ends. */
export function Recordings({taskId}: {taskId?: string}) {
  const [rows, setRows] = useState<RecordingSegment[]>([]),
    [playing, setPlaying] = useState('');
  const [error, setError] = useState('');
  const [merge, setMerge] = useState<MergeState | null>(null),
    [requesting, setRequesting] = useState(''),
    [requestError, setRequestError] = useState('');
  const shownTask = useRef(taskId);
  shownTask.current = taskId;
  useEffect(() => {
    setRows([]);
    setPlaying('');
    setError('');
    setMerge(null);
    setRequesting('');
    setRequestError('');
    if (!taskId) return;
    let stopped = false;
    const refresh = async () => {
      try {
        const [value, mergeValue] = await Promise.all([
          Api.get<RecordingSegment[]>(`/api/history/${taskId}/recordings`).catch(() => {
            throw new Error('Could not load recordings');
          }),
          Api.get<MergeState>(`/api/history/${taskId}/recordings/merged`),
        ]);
        if (!stopped) {
          setRows(value);
          setMerge(mergeValue);
          setError('');
        }
      } catch (e) {
        if (!stopped) setError((e as Error).message);
      }
    };
    refresh();
    const timer = setInterval(refresh, REFRESH_MS);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [taskId]);
  /** Start rendering one segment's videos, or all segments merged. */
  async function render(part: string) {
    const id = taskId;
    if (!id) return;
    setRequesting(part);
    setRequestError('');
    try {
      const path = part === MERGED_PART ? MERGED_PART : `${part}/styled`;
      const value = await Api.post<MergeState>(`/api/history/${id}/recordings/${path}`, undefined, null);
      if (shownTask.current !== id) return;
      if (part === MERGED_PART) setMerge(value);
      else setRows((all) => all.map((row) => (row.id === part ? {...row, render: value.render} : row)));
    } catch (e) {
      if (shownTask.current === id) setRequestError((e as Error).message);
    } finally {
      if (shownTask.current === id) setRequesting('');
    }
  }
  if (!taskId || (!rows.length && !error)) return null;
  const joined = merge?.current ? merge.merged : null;
  const mergeRender = inProgress(merge?.render);
  // A single segment's own videos are the task's; merging starts at two.
  const showMerge = !!merge && (merge.segments.full.length > 1 || !!mergeRender);
  const canMerge = !!merge && !merge.current && merge.segments.full.length > 1 && !mergeRender;
  return (
    <section className="recordings" aria-label="Task recordings">
      <div className="recordings-title">
        <Video size={16} />
        <strong>Task recordings</strong>
      </div>
      {error && <p role="alert">{error}</p>}
      {requestError && (
        <p className="recording-error" role="alert">
          {requestError}
        </p>
      )}
      {showMerge && merge && (
        <div className="recording-row recording-merged" role="group" aria-label="Merged video">
          <div>
            <strong>Merged video</strong>
            <span className={`recording-state ${joined ? 'saved' : ''}`}>
              {mergeRender ? (
                <RenderStage render={mergeRender} />
              ) : joined ? (
                `All ${joined.segments.full.length} segments`
              ) : (
                'Not merged'
              )}
            </span>
            {canMerge && (
              <button
                className="recording-merge"
                onClick={() => void render(MERGED_PART)}
                disabled={!!requesting || !!merge.pending}
              >
                {requesting === MERGED_PART ? <LoaderCircle className="spin" size={14} /> : <Layers size={14} />}
                {requesting === MERGED_PART ? 'Merging…' : `Merge: ${segmentCount(merge.segments.full.length)}`}
              </button>
            )}
          </div>
          <div className="recording-files">
            {joined ? (
              <VideoPair
                url={`/api/recordings/${taskId}/${MERGED_PART}`}
                video={joined}
                download={(label) => `Download merged ${label}`}
                onPlay={setPlaying}
              />
            ) : (
              !mergeRender && (
                <span>Rendered when the task ends: one full and one motion-only video of all segments</span>
              )
            )}
            {joined && !joined.motion && <span>No motion in any segment</span>}
          </div>
          {canMerge && merge.pending && (
            <p className="recording-note">Available after segment {Number(merge.pending)} finishes.</p>
          )}
          {joined &&
            [skipNote('Full video', joined.skipped.full), skipNote('Motion only', joined.skipped.motion)]
              .filter(Boolean)
              .map((note) => (
                <p className="recording-note" key={note}>
                  {note}
                </p>
              ))}
          {merge.render?.status === 'failed' && (
            <p className="recording-error" role="alert">
              Merge failed: {merge.render.error}
            </p>
          )}
        </div>
      )}
      {rows.map((row) => {
        const number = Number(row.id);
        const rendering = inProgress(row.render);
        const isSaved = row.status === 'saved';
        // A segment recorded before styled videos, or whose render failed, renders on request.
        const canRender = isSaved && !row.styled && !rendering;
        return (
          <div className="recording-row" key={row.id}>
            <div>
              <strong>Segment {number}</strong>
              <span className={`recording-state ${rendering && isSaved ? '' : row.status}`}>
                {rendering && isSaved ? <RenderStage render={rendering} /> : row.status}
              </span>
              {canRender && (
                <button className="recording-merge" onClick={() => void render(row.id)} disabled={!!requesting}>
                  {requesting === row.id ? <LoaderCircle className="spin" size={14} /> : <Film size={14} />}
                  {row.render ? 'Render again' : 'Render videos'}
                </button>
              )}
            </div>
            <div className="recording-files">
              {row.styled && (
                <VideoPair
                  url={`/api/recordings/${taskId}/${row.id}/styled`}
                  video={row.styled}
                  download={(label) => `Download ${label} segment ${number}`}
                  onPlay={setPlaying}
                />
              )}
              {row.styled && !row.styled.motion && <span>No motion in this segment</span>}
              {(isSaved || row.status === 'failed') && (
                <a
                  className="recording-source"
                  aria-label={`Download source files segment ${number}`}
                  title="Camera videos without captions, timeline and task records for re-editing"
                  href={`/api/recordings/${taskId}/${row.id}/source.zip`}
                >
                  <Archive size={15} />
                  Source files
                </a>
              )}
              {row.status === 'recording' && (
                <span>{duration(row.duration_s)} · Recording cameras and model actions</span>
              )}
              {row.status === 'processing' && <span>Preparing videos…</span>}
            </div>
            {row.error && (
              <p className="recording-error" role="alert">
                {row.error}
              </p>
            )}
            {isSaved && row.render?.status === 'failed' && (
              <p className="recording-error" role="alert">
                Rendering failed: {row.render.error}
              </p>
            )}
            {row.source_error && (
              <p className="recording-note">Some camera source files are missing: {row.source_error}</p>
            )}
            {!!row.dropped_samples && (
              <p className="recording-note">
                {row.dropped_samples} camera samples skipped under load; original timing preserved.
              </p>
            )}
          </div>
        );
      })}
      {playing && (
        <div className="recording-player">
          <button aria-label="Close video" onClick={() => setPlaying('')}>
            <X size={17} />
          </button>
          <video key={playing} src={playing} controls autoPlay playsInline />
        </div>
      )}
    </section>
  );
}
