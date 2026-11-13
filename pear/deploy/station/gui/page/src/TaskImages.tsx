import {useEffect, useRef, useState} from 'react';
import {ImagePlus, LoaderCircle, X} from 'lucide-react';
import {Api} from './api';
import type {SavedImage} from './protocol';
import './TaskImages.css';

const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const UPLOAD_TIMEOUT_MS = 65000;
const IMAGE_LIFETIME_S = 3600; // an uploaded image is deleted by the service after this long

type DraftImage = {
  key: string;
  file: File;
  url: string;
  status: 'uploading' | 'ready' | 'error';
  id?: string;
  created?: number;
  error?: string;
};
export function useTaskImages(clientId: string) {
  const [images, setImages] = useState<DraftImage[]>([]);
  const current = useRef<DraftImage[]>([]),
    requests = useRef(new Map<string, AbortController>());
  const update = (next: DraftImage[]) => {
    current.current = next;
    setImages(next);
  };
  const change = (key: string, patch: Partial<DraftImage>) =>
    update(current.current.map((im) => (im.key === key ? {...im, ...patch} : im)));
  const discard = (id: string) =>
    void Api.request(`/api/images/${id}?client_id=${clientId}`, {method: 'DELETE'}).catch(() => {});
  async function upload(im: DraftImage) {
    if (im.id) discard(im.id);
    requests.current.get(im.key)?.abort();
    const abort = new AbortController();
    requests.current.set(im.key, abort);
    change(im.key, {status: 'uploading', id: undefined, error: undefined});
    try {
      const form = new FormData();
      form.append('image', im.file);
      const value = await Api.request<{id: string; created: number}>(`/api/images?client_id=${clientId}`, {
        method: 'POST',
        form,
        timeout: UPLOAD_TIMEOUT_MS,
        signal: abort.signal,
      });
      if (!current.current.some((x) => x.key === im.key)) {
        discard(value.id);
        return;
      }
      change(im.key, {status: 'ready', id: value.id, created: value.created});
    } catch (e) {
      change(im.key, {
        status: 'error',
        error:
          e instanceof DOMException && e.name === 'AbortError' ? 'Upload interrupted; retry.' : (e as Error).message,
      });
    } finally {
      if (requests.current.get(im.key) === abort) requests.current.delete(im.key);
    }
  }
  function add(files: File[]) {
    if (current.current.length + files.length > MAX_IMAGES) return 'Choose up to four images.';
    if (files.some((f) => !IMAGE_TYPES.includes(f.type))) return 'Use JPEG, PNG or static WebP images.';
    if (files.some((f) => f.size > MAX_IMAGE_BYTES)) return 'Each image must be at most 10 MiB.';
    const added = files.map((file) => ({
      key: crypto.randomUUID(),
      file,
      url: URL.createObjectURL(file),
      status: 'uploading' as const,
    }));
    update([...current.current, ...added]);
    added.forEach((im) => void upload(im));
    return '';
  }
  function remove(key: string) {
    const im = current.current.find((x) => x.key === key);
    if (!im) return;
    update(current.current.filter((x) => x.key !== key));
    requests.current.get(key)?.abort();
    URL.revokeObjectURL(im.url);
    if (im.id) discard(im.id);
  }
  function reset() {
    current.current.forEach((im) => {
      requests.current.get(im.key)?.abort();
      URL.revokeObjectURL(im.url);
      if (im.id) discard(im.id);
    });
    update([]);
  }
  function invalidate() {
    update(
      current.current.map((im) => ({...im, status: 'error', error: 'Image expired or unavailable; retry upload.'})),
    );
  }
  useEffect(
    () => () => {
      requests.current.forEach((r) => r.abort());
      current.current.forEach((im) => URL.revokeObjectURL(im.url));
    },
    [],
  );
  return {
    images,
    add,
    remove,
    reset,
    invalidate,
    retry: (im: DraftImage) => void upload(im),
    ready: images.every((im) => im.status === 'ready'),
    ids: images.map((im) => im.id!),
    expired: () => current.current.some((im) => !im.created || Date.now() / 1000 - im.created >= IMAGE_LIFETIME_S),
  };
}

export function ImageAttachments({
  draft,
  disabled,
  onFiles,
  onPreview,
}: {
  draft: ReturnType<typeof useTaskImages>;
  disabled: boolean;
  onFiles: (files: File[]) => void;
  onPreview: (url: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="image-attachments">
      <input
        ref={input}
        type="file"
        aria-label="Choose task images"
        accept={IMAGE_TYPES.join(',')}
        multiple
        hidden
        disabled={disabled}
        onChange={(e) => {
          onFiles(Array.from(e.target.files || []));
          e.target.value = '';
        }}
      />
      <button
        type="button"
        className="attach-button"
        disabled={disabled || draft.images.length >= MAX_IMAGES}
        onClick={() => input.current?.click()}
      >
        <ImagePlus size={15} />
        Add images
      </button>
      {draft.images.length > 0 && (
        <div className="attachment-list">
          {draft.images.map((im, i) => (
            <div className="attachment" key={im.key}>
              <button
                className="attachment-preview"
                aria-label={`Preview task image ${i + 1}`}
                onClick={() => onPreview(im.url)}
              >
                <img src={im.url} alt={im.file.name} />
              </button>
              <button
                className="attachment-remove"
                aria-label={`Remove task image ${i + 1}`}
                disabled={disabled}
                onClick={() => draft.remove(im.key)}
              >
                <X size={13} />
              </button>
              <span className={`attachment-status ${im.status}`}>
                {im.status === 'uploading' ? (
                  <>
                    <LoaderCircle className="spin" size={12} />
                    Uploading…
                  </>
                ) : im.status === 'ready' ? (
                  'Ready'
                ) : (
                  <button disabled={disabled} onClick={() => draft.retry(im)}>
                    Retry upload
                  </button>
                )}
              </span>
              {im.error && (
                <span className="attachment-error" role="alert">
                  {im.error}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function SavedImages({images, onPreview}: {images?: SavedImage[]; onPreview: (url: string) => void}) {
  return images?.length ? (
    <div className="saved-images">
      {images.map((im, i) => (
        <button key={im.id} aria-label={`View task image ${i + 1}`} onClick={() => onPreview(im.url)}>
          <img src={im.url} alt={im.name} />
        </button>
      ))}
    </div>
  ) : null;
}

export function ImagePreview({url, onClose}: {url: string; onClose: () => void}) {
  useEffect(() => {
    const close = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [onClose]);
  return (
    <div
      className="image-preview-modal"
      role="dialog"
      aria-modal="true"
      aria-label="Task image preview"
      onClick={onClose}
    >
      <button autoFocus aria-label="Close image preview" onClick={onClose}>
        <X />
      </button>
      <img src={url} alt="Task image" onClick={(e) => e.stopPropagation()} />
    </div>
  );
}
