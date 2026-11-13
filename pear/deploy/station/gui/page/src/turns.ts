import type {Observation, PerformanceData, StreamEvent, TurnCapacity, TurnResult, TurnUsage} from './protocol';

/** The tokens the latest request put in the model's context, and the context's size. */
export type ContextUsage = {used?: number; limit?: number};

export type Turn = {
  id: string;
  number?: number;
  started?: number;
  phase?: string;
  phaseStarted?: number;
  observation?: Observation;
  reasoning: string;
  answer: string;
  answered?: number;
  result?: TurnResult;
  usage?: TurnUsage;
  capacity?: TurnCapacity;
  notice?: string;
  performance?: PerformanceData;
  inferenceStarted?: number;
  executionStarted?: number;
  executed?: number;
};

/** Folds the events of one task into its turns, in order of first appearance. */
export class Turns {
  static readonly ACTION_BLOCK = /```json\s*(\{[\s\S]*?\})\s*```/g; // a fenced json action, as the service reads it
  static readonly OPEN_FENCE = '```json'; // the start of an action block that is still streaming

  static fromEvents(events: StreamEvent[]): Turn[] {
    const turns: Turn[] = [];
    const byAttempt = new Map<string, Turn>();
    for (const event of events) {
      const id = event.data.attempt;
      if (!id) continue;

      let turn = byAttempt.get(id);
      if (!turn) {
        turn = {id, reasoning: '', answer: ''};
        byAttempt.set(id, turn);
        turns.push(turn);
      }
      Turns.apply(turn, event);
    }
    return turns;
  }

  /** The context of the latest turn that measured it; a separately measured context wins over the request's input. */
  static context(turns: Turn[]): ContextUsage {
    for (const t of [...turns].reverse()) {
      const measured = t.usage?.context;
      const used = measured ? measured.input : (t.usage?.input ?? t.capacity?.input);
      if (typeof used === 'number') {
        return {used, limit: measured ? measured.context_limit : (t.usage?.context_limit ?? t.capacity?.context_limit)};
      }
    }
    return {};
  }

  /** The tokens a turn generated: its call's count, else the streamed one; a lower bound when its usage is incomplete. */
  static generated(t: Turn): {count?: number; isLowerBound: boolean} {
    if (typeof t.usage?.output === 'number') {
      return {count: t.usage.output, isLowerBound: !!t.usage.partial && t.usage.final !== false};
    }
    const p = t.performance;
    return {count: p?.output_tokens ?? undefined, isLowerBound: !!(p?.final && p.partial)};
  }

  /** The prose of an answer, and the json of its last fenced action block, which the prose leaves out. */
  static split(answer: string): {text: string; block: string} {
    const block = [...answer.matchAll(Turns.ACTION_BLOCK)].pop();
    if (block) {
      const prose = answer.slice(0, block.index) + answer.slice(block.index + block[0].length);
      return {text: prose.trim(), block: block[1]};
    }
    const open = answer.indexOf(Turns.OPEN_FENCE);
    return {text: (open < 0 ? answer : answer.slice(0, open)).trim(), block: ''};
  }

  private static apply(turn: Turn, event: StreamEvent) {
    const {data, type, time} = event;
    if (data.turn_number) turn.number = data.turn_number;
    if (data.started_at) turn.started = data.started_at;
    if (data.phase) turn.phase = data.phase;
    if (type === 'turn_started' || type === 'phase') turn.phaseStarted = time;
    if (type === 'notice' || type === 'turn_end') turn.notice = data.message;
    if (type === 'observation') turn.observation = data as Observation;
    if (type === 'model_text' || type === 'model_output') {
      turn.reasoning = data.reasoning || '';
      turn.answer = data.answer || '';
    }
    if (type === 'model_output') turn.answered = time;
    if (type === 'reasoning_delta') turn.reasoning += data.text || '';
    if (type === 'answer_delta') turn.answer += data.text || '';
    if (type === 'reasoning') turn.reasoning = data.text || '';
    if (type === 'answer') turn.answer = data.text || '';
    if (type === 'execution') {
      turn.result = data as TurnResult;
      turn.executed = time;
    }
    if (type === 'usage') turn.usage = data as TurnUsage;
    if (type === 'performance') turn.performance = {...(data as PerformanceData), received_at: time};
    if (type === 'phase' && data.phase === 'Thinking') turn.inferenceStarted = time;
    if (type === 'phase' && data.phase === 'Executing') turn.executionStarted = time;
    if (type === 'capacity') turn.capacity = data as TurnCapacity;
  }
}
