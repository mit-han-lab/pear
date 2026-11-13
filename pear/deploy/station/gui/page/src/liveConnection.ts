import {useEffect, useRef, useState} from 'react';
import {Api, ApiError} from './api';
import {EventLog} from './eventLog';
import type {ServerState, StreamEvent} from './protocol';

const CONNECT_TIMEOUT_MS = 5000;
const STALE_MS = 3000; // no message for this long means the stream is lost
const HEARTBEAT_INTERVAL_MS = 1000;
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;
const RECONNECT_MAX_DOUBLINGS = 4;
const FRAME_REVOKE_DELAY_MS = 1000; // an old frame URL stays valid while the image element swaps

/** The authenticated, claimed live connection to the service: state, events and camera frames. */
export function useLiveConnection(clientId: string) {
  const [state, setState] = useState<ServerState | null>(null);
  const [connected, setConnected] = useState(false);
  const [authed, setAuthed] = useState(false);
  const [error, setError] = useState('');
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [frames, setFrames] = useState<Record<string, string>>({});
  const frameUrls = useRef<Record<string, string>>({});
  const stream = useRef<WebSocket | null>(null);
  const lastEvent = useRef(0);
  const lastReceived = useRef(0);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connecting = useRef(false);
  const mounted = useRef(true);
  const retryCount = useRef(0);
  const rememberedToken = useRef('');
  const serviceInstance = useRef('');
  const stateRef = useRef<ServerState | null>(null);
  stateRef.current = state;

  function addEvent(event: StreamEvent) {
    if (event.type === 'state') {
      setState(event.data as ServerState);
      return;
    }
    if (event.id <= lastEvent.current) return;
    lastEvent.current = event.id;
    setEvents((old) => EventLog.append(old, event));
  }

  function showFrame(bytes: Uint8Array) {
    const role = stateRef.current?.profile.cameras[bytes[0]]?.role;
    if (!role) return;
    const old = frameUrls.current[role];
    const url = URL.createObjectURL(new Blob([bytes.slice(1)], {type: 'image/jpeg'}));
    frameUrls.current[role] = url;
    setFrames((frames) => ({...frames, [role]: url}));
    if (old) setTimeout(() => URL.revokeObjectURL(old), FRAME_REVOKE_DELAY_MS);
  }

  function dropFrames() {
    Object.values(frameUrls.current).forEach(URL.revokeObjectURL);
    frameUrls.current = {};
  }

  function scheduleReconnect() {
    if (!mounted.current || reconnectTimer.current) return;
    const doublings = Math.min(retryCount.current++, RECONNECT_MAX_DOUBLINGS);
    const delay = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** doublings);
    reconnectTimer.current = setTimeout(() => {
      reconnectTimer.current = null;
      void connect();
    }, delay);
  }

  function detachStream() {
    const previous = stream.current;
    stream.current = null;
    if (poll.current) {
      clearInterval(poll.current);
      poll.current = null;
    }
    previous?.close();
  }

  async function fetchState(): Promise<ServerState> {
    try {
      return await Api.get<ServerState>('/api/state', CONNECT_TIMEOUT_MS);
    } catch (e) {
      if (!(e instanceof ApiError) || e.status !== 401 || !rememberedToken.current) throw e;
      await Api.post('/api/auth', {token: rememberedToken.current}, CONNECT_TIMEOUT_MS);
      return await Api.get<ServerState>('/api/state', CONNECT_TIMEOUT_MS);
    }
  }

  function openStream() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const source = new WebSocket(`${scheme}://${location.host}/api/live?after=${lastEvent.current}`);
    source.binaryType = 'arraybuffer';
    stream.current = source;
    lastReceived.current = Date.now();
    const heartbeat = () => source.send(JSON.stringify({type: 'heartbeat', client_id: clientId}));
    const lost = () => {
      if (stream.current !== source) return;
      detachStream();
      setConnected(false);
      scheduleReconnect();
    };
    source.onopen = () => {
      if (stream.current !== source) return;
      lastReceived.current = Date.now();
      retryCount.current = 0;
      setConnected(true);
      setError('');
      heartbeat();
    };
    source.onmessage = (message) => {
      if (stream.current !== source) return;
      lastReceived.current = Date.now();
      if (typeof message.data !== 'string') {
        showFrame(new Uint8Array(message.data));
        return;
      }
      try {
        addEvent(JSON.parse(message.data));
      } catch {}
    };
    source.onerror = lost;
    source.onclose = lost;
    poll.current = setInterval(() => {
      if (Date.now() - lastReceived.current > STALE_MS) lost();
      else if (source.readyState === WebSocket.OPEN) heartbeat();
    }, HEARTBEAT_INTERVAL_MS);
  }

  async function connect(token?: string) {
    if (token) rememberedToken.current = token;
    if (connecting.current || !mounted.current) return;
    connecting.current = true;
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    detachStream();
    setConnected(false);

    try {
      if (token) await Api.post('/api/auth', {token}, CONNECT_TIMEOUT_MS);
      const health = await Api.get<{instance: string}>('/health', CONNECT_TIMEOUT_MS);
      const fresh = await fetchState();
      if (!mounted.current) return;
      if (serviceInstance.current && serviceInstance.current !== health.instance) {
        lastEvent.current = 0;
        setEvents([]);
        setFrames({});
        dropFrames();
      }
      serviceInstance.current = health.instance;
      setState(fresh);
      setAuthed(true);
      await Api.post('/api/control/claim', {client_id: clientId}, CONNECT_TIMEOUT_MS);
      if (!mounted.current) return;
      openStream();
    } catch (e) {
      if (!mounted.current) return;
      setError(String(e instanceof Error ? e.message : e));
      setConnected(false);
      if (e instanceof ApiError && e.status === 401) {
        rememberedToken.current = '';
        setAuthed(false);
      } else scheduleReconnect();
    } finally {
      connecting.current = false;
    }
  }

  useEffect(() => {
    mounted.current = true;
    const token = new URLSearchParams(location.hash.slice(1)).get('token');
    if (token) window.history.replaceState(null, '', location.pathname);
    void connect(token || undefined);
    const offline = () => {
      detachStream();
      setConnected(false);
      scheduleReconnect();
    };
    const online = () => {
      if (stream.current?.readyState !== WebSocket.OPEN || Date.now() - lastReceived.current > STALE_MS) void connect();
    };
    const visible = () => {
      if (document.visibilityState === 'visible') online();
    };
    window.addEventListener('offline', offline);
    window.addEventListener('online', online);
    document.addEventListener('visibilitychange', visible);
    return () => {
      mounted.current = false;
      detachStream();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      window.removeEventListener('offline', offline);
      window.removeEventListener('online', online);
      document.removeEventListener('visibilitychange', visible);
      Object.values(frameUrls.current).forEach(URL.revokeObjectURL);
    };
  }, []);

  return {state, connected, authed, error, setError, events, clearEvents: () => setEvents([]), frames, connect};
}
