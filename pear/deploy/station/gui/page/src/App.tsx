import {useEffect, useState} from 'react';
import {AlertCircle, Cable, History, Layers, Radio, Square, X} from 'lucide-react';
import './theme.css';
import './App.css';
import {Api, ApiError} from './api';
import {useTagCounts} from './Tags';
import {ArmState} from './ArmState';
import {Camera, CAMERA_FRESH_AGE_S} from './Camera';
import {Composer, useDraft} from './Composer';
import {Conversation} from './Conversation';
import {ControlPanel} from './ControlPanel';
import {useHistory} from './History';
import {useLiveConnection} from './liveConnection';
import {CameraModal} from './CameraModal';
import {ConnectModal} from './ConnectModal';
import {ConfirmationDialog, type DialogKind} from './ConfirmationDialog';
import {ImagePreview} from './TaskImages';
import {Turns} from './turns';

const clientId = crypto.randomUUID();
const IMAGE_UNAVAILABLE = /image.*(expired|unavailable)/i;

export function App() {
  const live = useLiveConnection(clientId);
  const {state, connected, authed, error, setError, events} = live;
  const [pending, setPending] = useState('');
  const [tab, setTab] = useState('workspace');
  const [expandedCamera, setExpandedCamera] = useState<string | null>(null);
  const [dialog, setDialog] = useState<DialogKind | null>(null);
  const [homeSide, setHomeSide] = useState('');
  const [homeOpen, setHomeOpen] = useState(true);
  const [follow, setFollow] = useState(true);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const draft = useDraft(clientId);
  const {tagCounts, refreshTags} = useTagCounts(authed);
  const history = useHistory(refreshTags, setError);

  const mine = state?.owner === clientId;
  const enabled = state?.robot.status !== 'OFF' && !!state;
  const ready = state?.robot.status === 'READY';
  const busy = !!state?.busy;
  const task = state?.task;
  const stopped = !!state?.stop_latched;
  // The robot's description: cameras, names and Home choices all come from the service.
  const profile = state?.profile;
  const title = profile?.title ?? 'Robot';
  const cameras = profile?.cameras ?? [];
  const joints = profile?.joints ?? [];
  const homeSides = profile?.home_sides ?? [];
  const camerasReady =
    !state?.cameras.error &&
    cameras.length > 0 &&
    cameras.every((camera) => (state?.cameras.roles[camera.role]?.age ?? Infinity) < CAMERA_FRESH_AGE_S);
  const canRun = connected && mine && ready && !busy && !stopped && camerasReady;
  const isHistory = tab === 'history';
  const activeTask = isHistory ? history.opened : task;
  const activeEvents = (isHistory ? history.pastEvents : events).filter((e) => e.task_id === activeTask?.id);
  const turns = Turns.fromEvents(activeEvents);
  // The control panel reports the live task, also while History shows another.
  const liveTurns = isHistory ? Turns.fromEvents(events.filter((e) => e.task_id === task?.id)) : turns;
  const observed = isHistory ? undefined : [...turns].reverse().find((t) => t.observation)?.observation?.state;
  useEffect(() => {
    document.title = `${title} Control`;
  }, [title]);

  async function command(name: string, data: Record<string, unknown> = {}): Promise<boolean> {
    if (!connected && name !== 'stop') {
      setError('Connection lost. Reconnect before continuing.');
      return false;
    }
    setPending(name);
    setError('');
    try {
      await Api.post('/api/command', {
        client_id: clientId,
        command_id: name === 'start' ? (draft.startId.current ??= crypto.randomUUID()) : crypto.randomUUID(),
        command: name,
        data,
      });
      if (name === 'clear') {
        draft.clear();
        live.clearEvents();
      }
      if (name === 'start') setFollow(true);
      return true;
    } catch (e) {
      setError((e as Error).message);
      if (name === 'start' && e instanceof ApiError) {
        draft.startId.current = null;
        if (IMAGE_UNAVAILABLE.test(e.message)) draft.images.invalidate();
      }
      return false;
    } finally {
      setPending('');
    }
  }

  function showHistory() {
    setTab('history');
    void history.load();
  }

  function openDialog(kind: DialogKind, openGrippers: boolean) {
    setHomeSide(homeSides[0] || '');
    setHomeOpen(openGrippers);
    setDialog(kind);
  }

  const cameraFeed = (role: string, cameraLabel: string) => (
    <Camera
      key={role}
      role={role}
      label={cameraLabel}
      state={state}
      connected={connected}
      frame={live.frames[role]}
      onExpand={setExpandedCamera}
    />
  );
  const expanded = cameras.find((camera) => camera.role === expandedCamera);

  return (
    <div className={`app-shell ${cameras.length === 1 ? 'single-camera' : ''}`}>
      {previewImage && <ImagePreview url={previewImage} onClose={() => setPreviewImage(null)} />}
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">{title[0]}</span>
          <span>
            {title}
            <span className="brand-light">Control</span>
          </span>
        </a>
        <div className="header-spacer" />
        <span className={`connection ${connected ? 'ok' : ''}`}>
          <i />
          {connected ? 'Connected' : 'Disconnected'}
        </span>
        <span className="environment">{state?.test_mode ? 'TEST MODE · SIMULATED' : 'LIVE ROBOT'}</span>
        <button className="stop-button" onClick={() => command('stop')} disabled={!authed}>
          <Square size={15} fill="currentColor" /> STOP MOTION
        </button>
      </header>
      <main>
        <div className="workspace-heading">
          <div>
            <h1>
              A clear view. <span>Every step.</span>
            </h1>
          </div>
          <div className="view-tabs">
            <button className={tab === 'workspace' ? 'selected' : ''} onClick={() => setTab('workspace')}>
              <Layers size={15} />
              Workspace
            </button>
            <button className={isHistory ? 'selected' : ''} onClick={showHistory}>
              <History size={15} />
              History
            </button>
          </div>
        </div>
        <section className="camera-grid" aria-label="Camera feeds">
          {cameras.map((camera) => cameraFeed(camera.role, camera.label))}
          {joints.length > 0 && (
            <ArmState
              joints={joints}
              robot={state?.robot}
              observed={observed}
              format={isHistory ? history.opened?.action_format : (task?.action_format ?? state?.action_format)}
            />
          )}
        </section>
        <div className="feed-note">
          <Radio size={12} />
          {state?.backend === 'fake'
            ? 'TEST MODE · Simulated camera sources · No physical robot connected'
            : 'Live preview · Model observations are captured separately for each step'}
          <span>LIVE VIEW</span>
        </div>
        {(error || state?.error) && (
          <div className="alert" role="alert">
            <AlertCircle size={17} />
            <span>{error || state?.error}</span>
            <button aria-label="Dismiss error" onClick={() => setError('')}>
              <X size={15} />
            </button>
          </div>
        )}
        {!connected && authed && (
          <div className="alert">
            <Cable size={17} />
            <span>
              Control connection lost. Reconnecting automatically. Running tasks pause; resume explicitly after
              reconnection.
            </span>
            <button onClick={() => live.connect()}>Reconnect</button>
          </div>
        )}
        <div className="lower-grid">
          <section className="conversation-panel">
            <Conversation
              isHistory={isHistory}
              state={state}
              connected={connected}
              mine={mine}
              pending={pending}
              activeTask={activeTask}
              activeEvents={activeEvents}
              turns={turns}
              history={history}
              tagCounts={tagCounts}
              follow={follow}
              events={events}
              onFollow={setFollow}
              onClear={() => command('clear')}
              onPreview={setPreviewImage}
            />
            {!isHistory && (
              <Composer
                state={state}
                connected={connected}
                mine={mine}
                pending={pending}
                enabled={enabled}
                canRun={canRun}
                draft={draft}
                tagCounts={tagCounts}
                follow={follow}
                onFollow={setFollow}
                command={command}
                onError={setError}
                onTagsChanged={refreshTags}
                onAnnotationsSaved={history.saveAnnotations}
                onPreview={setPreviewImage}
              />
            )}
          </section>
          <ControlPanel
            state={state}
            context={Turns.context(liveTurns)}
            connected={connected}
            mine={mine}
            pending={pending}
            camerasReady={camerasReady}
            command={command}
            onHome={(openGrippers) => openDialog('home', openGrippers)}
            onDisable={() => setDialog('disable')}
            onReconnect={() => live.connect()}
            onError={setError}
          />
        </div>
      </main>
      {!authed && <ConnectModal error={error} onConnect={(token) => live.connect(token)} />}
      {expanded && (
        <CameraModal
          role={expanded.role}
          label={expanded.label}
          state={state}
          connected={connected}
          frame={live.frames[expanded.role]}
          onClose={() => setExpandedCamera(null)}
        />
      )}
      <ConfirmationDialog
        kind={dialog}
        state={state}
        command={command}
        homeOpen={homeOpen}
        homeSide={homeSide}
        onHomeSide={setHomeSide}
        onClose={() => setDialog(null)}
      />
    </div>
  );
}
