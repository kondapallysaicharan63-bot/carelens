import { describe, expect, it, vi } from 'vitest';
import { buildConstraints, openStream } from '../src/capture/session';
import { Consent, DISCLAIMER } from '../src/ui/consent';

const md = () => ({ getUserMedia: vi.fn().mockResolvedValue({} as MediaStream) });

describe('consent state machine', () => {
  it('disclaimer is verbatim', () => {
    expect(DISCLAIMER).toBe(
      'Not a medical device. For demonstration and wellbeing trends only. Do not use for diagnosis or emergencies; call emergency services if you feel unwell.',
    );
  });
  it('does not call getUserMedia before Start', async () => {
    const c = new Consent();
    const m = md();
    await expect(openStream(c, m)).rejects.toThrow(/Start/);
    expect(m.getUserMedia).not.toHaveBeenCalled();
  });
  it('cannot start with both toggles off', () => {
    const c = new Consent();
    c.setCamera(false);
    c.setMic(false);
    expect(c.canStart()).toBe(false);
    expect(c.start()).toBe(false);
  });
  it('camera only requests video only', async () => {
    const c = new Consent();
    c.setCamera(true);
    c.setMic(false);
    c.start();
    const m = md();
    await openStream(c, m);
    const arg = m.getUserMedia.mock.calls[0]![0] as MediaStreamConstraints;
    expect(arg.audio).toBe(false);
    expect(arg.video).toBeTruthy();
  });
  it('mic only requests audio only', async () => {
    const c = new Consent();
    c.setCamera(false);
    c.setMic(true);
    c.start();
    const m = md();
    await openStream(c, m);
    const arg = m.getUserMedia.mock.calls[0]![0] as MediaStreamConstraints;
    expect(arg.video).toBe(false);
    expect(arg.audio).toBeTruthy();
  });
  it('both on requests both; toggles are locked after Start', () => {
    const c = new Consent();
    c.setCamera(true);
    c.setMic(true);
    c.start();
    c.setMic(false);
    expect(c.state.mic).toBe(true);
    const k = buildConstraints(c.state)!;
    expect(k.video).toBeTruthy();
    expect(k.audio).toBeTruthy();
  });
  it('stop returns to unconsented', async () => {
    const c = new Consent();
    c.start();
    c.stop();
    await expect(openStream(c, md())).rejects.toThrow();
  });
  it('picks device by deviceId', () => {
    const k = buildConstraints({ camera: true, mic: false }, { videoDeviceId: 'abc' })!;
    expect((k.video as MediaTrackConstraints).deviceId).toEqual({ exact: 'abc' });
  });
});
