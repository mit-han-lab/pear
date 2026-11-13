import type {EventData, StreamEvent} from './protocol';

const TEXT_KINDS = ['reasoning_delta', 'answer_delta', 'reasoning', 'answer', 'model_text'];

/** The events kept in the browser: output of one turn coalesces, and only the latest are kept. */
export class EventLog {
  static readonly MAX_EVENTS = 1200;

  static append(old: StreamEvent[], event: StreamEvent): StreamEvent[] {
    const kind = event.type;
    if (kind === 'performance' || kind === 'usage') {
      const kept = old.filter(
        (e) => !(e.type === kind && e.task_id === event.task_id && e.data.attempt === event.data.attempt),
      );
      return [...kept, event].slice(-EventLog.MAX_EVENTS);
    }
    if (TEXT_KINDS.includes(kind)) return EventLog.appendText(old, event);
    return [...old, event].slice(-EventLog.MAX_EVENTS);
  }

  private static appendText(old: StreamEvent[], event: StreamEvent): StreamEvent[] {
    const kind = event.type;
    const index = old.findIndex(
      (e) => e.type === 'model_text' && e.task_id === event.task_id && e.data.attempt === event.data.attempt,
    );
    const existing: EventData = index >= 0 ? old[index].data : {reasoning: '', answer: ''};
    const data: EventData = {
      ...existing,
      ...(kind === 'model_text' ? event.data : {}),
      attempt: event.data.attempt,
      turn_number: event.data.turn_number,
      started_at: event.data.started_at,
      phase: event.data.phase,
    };
    if (event.data.reasoning_kind) data.reasoning_kind = event.data.reasoning_kind;
    if (kind.startsWith('reasoning')) {
      data.reasoning = (kind.endsWith('_delta') ? existing.reasoning : '') + (event.data.text || '');
    }
    if (kind.startsWith('answer'))
      data.answer = (kind.endsWith('_delta') ? existing.answer : '') + (event.data.text || '');

    const merged = {...event, type: 'model_text', data};
    const next = [...old];
    if (index >= 0) next[index] = merged;
    else next.push(merged);
    return next.slice(-EventLog.MAX_EVENTS);
  }
}
