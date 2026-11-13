export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

type RequestOptions = {
  method?: string;
  json?: unknown;
  form?: FormData;
  timeout?: number | null; // null waits as long as the service takes
  signal?: AbortSignal;
};

/** The only place the browser calls `fetch`; every request carries the session cookie. */
export class Api {
  static readonly TIMEOUT_MS = 20000;

  static get<T>(path: string, timeout?: number | null): Promise<T> {
    return Api.request<T>(path, {timeout});
  }

  static post<T>(path: string, json?: unknown, timeout?: number | null): Promise<T> {
    return Api.request<T>(path, {method: 'POST', json, timeout});
  }

  static async request<T>(path: string, request: RequestOptions = {}): Promise<T> {
    const abort = new AbortController();
    const timeout = request.timeout === undefined ? Api.TIMEOUT_MS : request.timeout;
    const timer = timeout === null ? null : setTimeout(() => abort.abort(), timeout);
    const cancel = () => abort.abort();
    if (request.signal?.aborted) cancel();
    request.signal?.addEventListener('abort', cancel, {once: true});
    const init: RequestInit = {credentials: 'same-origin', signal: abort.signal, method: request.method};
    if (request.json !== undefined) {
      init.headers = {'Content-Type': 'application/json'};
      init.body = JSON.stringify(request.json);
    }
    if (request.form) init.body = request.form;

    try {
      const response = await fetch(path, init);
      if (!response.ok) throw new ApiError(await Api.errorText(response), response.status);
      return await response.json();
    } finally {
      if (timer !== null) clearTimeout(timer);
      request.signal?.removeEventListener('abort', cancel);
    }
  }

  private static async errorText(response: Response): Promise<string> {
    try {
      const message = JSON.parse(await response.text()).error;
      if (message) return message;
    } catch {}
    return `Request failed (${response.status})`;
  }
}
