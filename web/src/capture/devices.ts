export type Facing = 'user' | 'environment';

export interface CaptureChoice {
  videoDeviceId?: string;
  audioDeviceId?: string;
  facing?: Facing;
}

/** Torch toggle is offered only when the track reports the capability. */
export function supportsTorch(track: Pick<MediaStreamTrack, 'getCapabilities'> | null | undefined): boolean {
  if (!track || typeof track.getCapabilities !== 'function') return false;
  try {
    return (track.getCapabilities() as { torch?: boolean }).torch === true;
  } catch {
    return false;
  }
}

export async function setTorch(
  track: Pick<MediaStreamTrack, 'getCapabilities' | 'applyConstraints'>,
  on: boolean,
): Promise<boolean> {
  if (!supportsTorch(track)) return false;
  await track.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] });
  return true;
}

export async function listDevices(md: Pick<MediaDevices, 'enumerateDevices'>) {
  const all = await md.enumerateDevices();
  return {
    cameras: all.filter((d) => d.kind === 'videoinput'),
    mics: all.filter((d) => d.kind === 'audioinput'),
  };
}

/** The next camera after `currentId`, wrapping around; undefined if there is no other. */
export function nextDeviceId(devices: { deviceId: string }[], currentId: string | undefined): string | undefined {
  if (devices.length < 2) return undefined;
  const i = devices.findIndex((d) => d.deviceId === currentId);
  return devices[(i + 1) % devices.length]?.deviceId;
}

/** Human-readable message for getUserMedia failures. */
export function describeMediaError(err: unknown): string {
  const name = (err as { name?: string })?.name ?? '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Permission was denied. Allow camera/microphone access in your browser’s site settings, then press Start again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No matching camera or microphone was found. Check that one is connected and not disabled.';
    case 'NotReadableError':
    case 'AbortError':
      return 'The device could not be opened. Another app or tab may be using it. Close it and try again, or choose another device.';
    default:
      return 'Could not start the camera or microphone. ' + (err instanceof Error ? err.message : '');
  }
}
