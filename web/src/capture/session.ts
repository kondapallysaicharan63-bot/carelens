import type { Consent } from '../ui/consent';
import type { CaptureChoice } from './devices';

export function buildConstraints(
  flags: { camera: boolean; mic: boolean },
  choice: CaptureChoice = {},
): MediaStreamConstraints | null {
  if (!flags.camera && !flags.mic) return null;
  const c: MediaStreamConstraints = { video: false, audio: false };
  if (flags.camera) {
    c.video = choice.videoDeviceId
      ? { deviceId: { exact: choice.videoDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: { ideal: choice.facing ?? 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } };
  }
  if (flags.mic) {
    // Raw-ish audio: no processing that would distort sound-event classification.
    c.audio = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      ...(choice.audioDeviceId ? { deviceId: { exact: choice.audioDeviceId } } : {}),
    };
  }
  return c;
}

/** Opens a stream. Refuses unless the user has pressed Start, and honours the consent flags. */
export async function openStream(
  consent: Consent,
  md: Pick<MediaDevices, 'getUserMedia'>,
  choice: CaptureChoice = {},
): Promise<MediaStream> {
  if (!consent.state.started) throw new Error('Consent not given: press Start first.');
  const constraints = buildConstraints(consent.state, choice);
  if (!constraints) throw new Error('Neither camera nor microphone is enabled.');
  return md.getUserMedia(constraints);
}

export function stopStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((t) => t.stop());
}

const STALE_ID_ERRORS = ['NotFoundError', 'OverconstrainedError'];

/**
 * Opens a stream; if a saved deviceId is stale (NotFound/Overconstrained), drops the ids and
 * retries once. `onDropped` lets the caller clear the persisted keys.
 */
export async function openStreamWithFallback(
  consent: Consent,
  md: Pick<MediaDevices, 'getUserMedia'>,
  choice: CaptureChoice,
  onDropped?: (dropped: ('videoDeviceId' | 'audioDeviceId')[]) => void,
): Promise<{ stream: MediaStream; choice: CaptureChoice }> {
  try {
    return { stream: await openStream(consent, md, choice), choice };
  } catch (err) {
    const name = (err as { name?: string })?.name ?? '';
    const dropped: ('videoDeviceId' | 'audioDeviceId')[] = [];
    if (consent.state.camera && choice.videoDeviceId) dropped.push('videoDeviceId');
    if (consent.state.mic && choice.audioDeviceId) dropped.push('audioDeviceId');
    if (!STALE_ID_ERRORS.includes(name) || dropped.length === 0) throw err;
    const cleaned: CaptureChoice = { ...choice };
    for (const k of dropped) delete cleaned[k];
    onDropped?.(dropped);
    return { stream: await openStream(consent, md, cleaned), choice: cleaned };
  }
}
