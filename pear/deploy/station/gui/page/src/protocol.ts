type CameraProfile = {role: string; label: string};

type RobotProfile = {
  title: string;
  cameras: CameraProfile[];
  home_sides: string[];
  home_keep_grippers: boolean;
  joints: string[];
};

type ModelChoice = {id: string; label: string; action_format?: string};

export type RobotStatus = {
  status?: string;
  arms?: Record<string, {position?: number[]}>;
  recovery?: {phase?: string};
};

type CameraStatus = {
  error: string | null;
  errors?: Record<string, string>; // absent while no camera preview runs
  roles: Record<string, {age: number; frame?: number}>;
};

export type SavedImage = {id: string; url: string; name: string};

export type Task = {
  id: string;
  instruction: string;
  created: number;
  model: string;
  model_label?: string;
  backend?: string;
  action_format?: string;
  images?: SavedImage[];
  tags?: string[];
};

export type HomePlan = {
  plan_id: string;
  open_grippers?: boolean;
  active_sides?: string[];
  operation_seconds_estimate?: number;
};

type RecordingState = {enabled: boolean; active: {status?: string} | null};

export type PerformanceData = {
  output_tokens?: number | null;
  client_decode_throughput_tps?: number | null;
  elapsed_s?: number | null;
  received_at?: number;
  final?: boolean;
  status?: string;
  partial?: boolean;
};

export type ServerState = {
  phase: string;
  backend: string;
  profile: RobotProfile;
  model: string;
  models: ModelChoice[];
  action_format: string;
  test_mode: boolean;
  active_turn: {attempt?: string} | null;
  robot: RobotStatus;
  model_health: {status: string};
  supports_reference_images: boolean;
  busy: boolean;
  stop_latched: boolean;
  error: string | null;
  task: Task | null;
  owner: string | null;
  home_plan: HomePlan | null;
  performance: PerformanceData | null;
  inference_started_at: number | null;
  recording: RecordingState;
  cameras: CameraStatus;
};

export type EventData = {
  attempt?: string;
  turn_number?: number;
  started_at?: number;
  phase?: string;
  reasoning_kind?: string;
  text?: string;
  message?: string;
  reason?: string;
  reasoning?: string;
  answer?: string;
  [key: string]: unknown;
};

export type StreamEvent = {id: number; type: string; data: EventData; task_id?: string; time?: number};

export type Observation = {images?: Record<string, string>; state?: Record<string, Record<string, unknown>>};

export type TurnResult = {
  status?: string;
  settled?: boolean;
  reached?: boolean;
  gripper_outcome?: string | null;
  gripper_outcomes?: Record<string, string>;
  error?: {code?: string; message?: string} | null;
};

export type TurnUsage = {
  context?: {input?: number; context_limit?: number};
  input?: number;
  output?: number;
  final?: boolean;
  partial?: boolean;
  context_limit?: number;
};

export type TurnCapacity = {input?: number; context_limit?: number};

export type TagCount = {tag: string; count: number};

export type Annotation = {tags: string[]};

export type Skip = {id: string; reason: string};

export type RecordingSegment = {
  id: string;
  status: string;
  duration_s?: number;
  error?: string;
  dropped_samples?: number;
  source_error?: string | null;
  /** The segment's styled videos once rendered. */
  styled?: MergedRecording | null;
  render?: VideoRender | null;
};

/** A render of styled videos still under way, or the last one, which failed. */
export type VideoRender = {
  status: 'waiting' | 'queued' | 'rendering' | 'failed';
  progress: number | null;
  error: string | null;
};

export type MergedRecording = {
  created: number;
  full: string;
  motion: string | null;
  duration_s: number;
  motion_duration_s: number;
  segments: {full: {id: string}[]; motion: {id: string}[]};
  skipped: {full: Skip[]; motion: Skip[]};
};

export type MergeState = {
  /** Segments a merge would join now; `current` says whether `merged` already joins exactly these. */
  segments: {full: string[]; motion: string[]};
  pending: string | null;
  current: boolean;
  merged: MergedRecording | null;
  render?: VideoRender | null;
};
