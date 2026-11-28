import { detectBlackVideo } from '../capture/blackframe';
import { describeMediaError, listDevices, nextDeviceId, setTorch, supportsTorch, type CaptureChoice, type Facing } from '../capture/devices';
import { onVideoFrame } from '../capture/frames';
import { PresenceSession } from '../capture/presence';
import { createPoseSmoke } from '../capture/pose';
import { openStreamWithFallback, stopStream } from '../capture/session';
import { safeGet, safeRemove, safeSet } from '../storage';
import { makeCard, TIER1, TIER2, type Card } from './cards';
import { buildFingertipView, FingertipSession } from './fingertip';
import { qualityMeter } from './quality';
import { Consent, DISCLAIMER } from './consent';
import { el } from './dom';
import { NOT_OFFERED } from './notOffered';

type Theme = 'auto' | 'light' | 'dark';
type Mode = 'standard' | 'fingertip';

export function mountApp(root: HTMLElement): void {
  const consent = new Consent();
  let stream: MediaStream | null = null;
  let choice: CaptureChoice = {};
  let presenceSession: PresenceSession | null = null;
  let fingertipSession: FingertipSession | null = null;
  let mode: Mode = 'standard';
  // The user's standard-mode toggles, kept while fingertip mode forces camera-only.
  let stdCamera = consent.state.camera;
  let stdMic = consent.state.mic;
  let gen = 0; // capture generation: bumped on every Stop/Start so stale async work can bail out

  const savedCam = safeGet('videoDeviceId');
  if (savedCam) choice.videoDeviceId = savedCam;
  const savedMic = safeGet('audioDeviceId');
  if (savedMic) choice.audioDeviceId = savedMic;

  // ---- theme ----
  const applyTheme = (t: Theme) => {
    if (t === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
  };
  let theme = (safeGet('theme') as Theme | null) ?? 'auto';
  applyTheme(theme);
  const themeBtn = el('button', { type: 'button', class: 'btn ghost', 'aria-label': 'Change colour theme' });
  const themeLabel = () => (themeBtn.textContent = `Theme: ${theme}`);
  themeLabel();
  themeBtn.addEventListener('click', () => {
    theme = theme === 'auto' ? 'light' : theme === 'light' ? 'dark' : 'auto';
    safeSet('theme', theme);
    applyTheme(theme);
    themeLabel();
  });

  const header = el(
    'header',
    { class: 'site-header' },
    el('div', { class: 'wrap bar' }, el('h1', {}, 'CareLens'), el('nav', {}, el('a', { href: '#not-offered' }, 'Not offered'), themeBtn)),
  );
  const main = el('main', { class: 'wrap' });
  const footer = el('footer', { class: 'wrap foot' }, el('p', {}, 'Not a medical device. Nothing leaves your device: no uploads, no accounts, no tracking.'));
  root.replaceChildren(header, main, footer);

  const notOffered = el('section', { id: 'not-offered', class: 'panel' }, el('h2', {}, 'Why some measurements aren’t offered'));
  const dl = el('dl', { class: 'no-list' });
  for (const n of NOT_OFFERED) dl.append(el('dt', {}, n.name), el('dd', {}, n.reason));
  notOffered.append(dl);

  // ---- consent screen ----
  function showConsent(error?: string): void {
    stopAll();
    consent.stop();
    const cam = el('input', { type: 'checkbox', id: 'consent-camera' });
    const mic = el('input', { type: 'checkbox', id: 'consent-mic' });
    cam.checked = stdCamera;
    mic.checked = stdMic;
    const start = el('button', { type: 'button', class: 'btn primary', id: 'start' }, 'Start');
    const micRow = el('label', { class: 'toggle' }, mic, el('span', {}, 'Use microphone'));
    const modeStd = el('input', { type: 'radio', name: 'mode', id: 'mode-standard', value: 'standard' });
    const modeFp = el('input', { type: 'radio', name: 'mode', id: 'mode-fingertip', value: 'fingertip' });
    modeStd.checked = mode === 'standard';
    modeFp.checked = mode === 'fingertip';
    const sync = () => {
      mode = modeFp.checked ? 'fingertip' : 'standard';
      if (mode === 'fingertip') {
        mic.checked = false; // fingertip mode needs the camera only (the standard choices are kept)
        cam.checked = true;
      } else {
        cam.checked = stdCamera;
        mic.checked = stdMic;
      }
      micRow.hidden = mode === 'fingertip';
      cam.disabled = mode === 'fingertip';
      start.textContent = mode === 'fingertip' ? 'Start fingertip heart rate' : 'Start';
      consent.setCamera(cam.checked);
      consent.setMic(mic.checked);
      start.disabled = !consent.canStart();
    };
    modeStd.addEventListener('change', sync);
    modeFp.addEventListener('change', sync);
    cam.addEventListener('change', () => {
      if (mode === 'standard') stdCamera = cam.checked;
      sync();
    });
    mic.addEventListener('change', () => {
      if (mode === 'standard') stdMic = mic.checked;
      sync();
    });
    start.addEventListener('click', () => {
      sync();
      if (consent.start()) void (mode === 'fingertip' ? beginFingertip() : beginCapture());
    });
    sync();
    const errBox = el('p', { class: 'error', role: 'alert' }, error ?? '');
    errBox.hidden = !error;
    main.replaceChildren(
      el(
        'section',
        { class: 'panel consent' },
        el('h2', {}, 'Before you start'),
        el('p', { class: 'disclaimer', id: 'disclaimer' }, DISCLAIMER),
        el('p', {}, 'Everything is processed on this device. Video and sound are never uploaded or saved. Choose what to allow; you can change this each time.'),
        el(
          'fieldset',
          { class: 'modes' },
          el('legend', {}, 'What to measure'),
          el('label', { class: 'toggle' }, modeStd, el('span', {}, 'Camera and microphone checks')),
          el('label', { class: 'toggle' }, modeFp, el('span', {}, 'Heart rate (fingertip): back camera and flash')),
        ),
        el('label', { class: 'toggle' }, cam, el('span', {}, 'Use camera')),
        micRow,
        errBox,
        start,
      ),
      resultsSections(),
      notOffered,
    );
  }

  function resultsSections(presence?: Card): HTMLElement {
    const wrap = el('div', {});
    const r = el('section', { class: 'results', 'aria-labelledby': 'h-results' }, el('h2', { id: 'h-results' }, 'Results'), el('p', { class: 'sub' }, 'Tier 1: wellbeing trends with a signal-quality meter.'));
    const grid = el('div', { class: 'grid' });
    grid.append(presence ? presence.root : makeCard('Person in view', 1, 'Starts when the camera is on.').root);
    for (const [t, n] of TIER1) grid.append(makeCard(t, 1, n).root);
    r.append(grid);
    const e = el('section', { class: 'results experimental', 'aria-labelledby': 'h-exp' }, el('h2', { id: 'h-exp' }, 'Experimental'), el('p', { class: 'sub' }, 'Tier 2: early, less reliable features. Treat as curiosity, not information.'));
    const g2 = el('div', { class: 'grid' });
    for (const [t, n] of TIER2) g2.append(makeCard(t, 2, n).root);
    e.append(g2);
    wrap.append(r, e);
    return wrap;
  }

  // ---- live ----
  function stopAll(): void {
    gen++;
    presenceSession?.end();
    presenceSession = null;
    fingertipSession?.end();
    fingertipSession = null;
    stopStream(stream);
    stream = null;
  }

  async function beginCapture(): Promise<void> {
    const g = ++gen;
    const presence = makeCard('Person in view', 1, 'Pose model runs on this device (smoke test).', consent.state.camera ? 'Loading pose model…' : 'Camera is off');
    const video = el('video', { playsinline: '', muted: '', autoplay: '', 'aria-label': 'Camera preview' });
    video.muted = true;
    video.playsInline = true;
    const status = el('p', { class: 'status', role: 'status' }, 'Opening devices…');
    const warn = el('p', { class: 'error', role: 'alert' });
    warn.hidden = true;
    const camSel = el('select', { 'aria-label': 'Camera' });
    const micSel = el('select', { 'aria-label': 'Microphone' });
    const facingSel = el('select', { 'aria-label': 'Camera facing' }, el('option', { value: 'user' }, 'Front camera'), el('option', { value: 'environment' }, 'Back camera'));
    const torchBtn = el('button', { type: 'button', class: 'btn', 'aria-pressed': 'false' }, 'Torch off');
    torchBtn.hidden = true;
    const nextBtn = el('button', { type: 'button', class: 'btn' }, 'Try next camera');
    nextBtn.hidden = true;
    const stopBtn = el('button', { type: 'button', class: 'btn' }, 'Stop');
    stopBtn.addEventListener('click', () => showConsent());
    const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;

    const controls = el('div', { class: 'controls' });
    if (consent.state.camera) controls.append(el('label', {}, 'Camera', camSel));
    if (consent.state.camera && coarse) controls.append(el('label', {}, 'Facing', facingSel));
    if (consent.state.mic) controls.append(el('label', {}, 'Microphone', micSel));
    controls.append(el('div', { class: 'btnrow' }, torchBtn, nextBtn, stopBtn));

    const live = el('section', { class: 'panel live' }, el('h2', {}, 'Live'), status, warn);
    if (consent.state.camera) live.append(el('div', { class: 'preview' }, video));
    live.append(controls);
    main.replaceChildren(live, resultsSections(presence), notOffered);

    let torchOn = false;
    const openWith = async (c: CaptureChoice): Promise<boolean> => {
      stopStream(stream);
      stream = null;
      warn.hidden = true;
      nextBtn.hidden = true;
      torchOn = false;
      torchBtn.textContent = 'Torch off';
      torchBtn.setAttribute('aria-pressed', 'false');
      let opened: MediaStream;
      try {
        const r = await openStreamWithFallback(consent, navigator.mediaDevices, c, (keys) => keys.forEach((k) => safeRemove(k)));
        opened = r.stream;
        c = r.choice;
      } catch (err) {
        if (g === gen) showConsent(describeMediaError(err));
        return false;
      }
      if (g !== gen) {
        stopStream(opened); // Stop was pressed while the device was opening
        return false;
      }
      stream = opened;
      choice = c;
      const vtrack = stream.getVideoTracks()[0];
      const atrack = stream.getAudioTracks()[0];
      const devs = await listDevices(navigator.mediaDevices);
      if (g !== gen) return false;
      fill(camSel, devs.cameras, vtrack?.getSettings().deviceId, 'Camera');
      fill(micSel, devs.mics, atrack?.getSettings().deviceId, 'Microphone');
      torchBtn.hidden = !supportsTorch(vtrack);
      if (vtrack) {
        video.srcObject = new MediaStream([vtrack]);
        try {
          await video.play();
        } catch {
          /* autoplay may need a gesture on some browsers; Start was a gesture so this is rare */
        }
        if (g !== gen) return false;
        status.textContent = `Camera: ${vtrack.label || 'unnamed camera'}`;
        const black = await detectBlackVideo(video);
        if (g !== gen) return false;
        if (black) {
          warn.textContent = 'This camera is sending black frames — try another camera.';
          warn.hidden = false;
          nextBtn.hidden = devs.cameras.length < 2;
        }
      } else {
        status.textContent = 'Microphone on (no camera).';
      }
      presenceSession?.restart(); // re-bind the frame loop to the (new) stream
      return true;
    };

    camSel.addEventListener('change', () => {
      safeSet('videoDeviceId', camSel.value);
      void openWith({ ...choice, videoDeviceId: camSel.value });
    });
    micSel.addEventListener('change', () => {
      safeSet('audioDeviceId', micSel.value);
      void openWith({ ...choice, audioDeviceId: micSel.value });
    });
    facingSel.addEventListener('change', () => {
      const { videoDeviceId: _drop, ...rest } = choice;
      safeRemove('videoDeviceId'); // facing choice supersedes a remembered device
      void openWith({ ...rest, facing: facingSel.value as Facing });
    });
    nextBtn.addEventListener('click', async () => {
      const devs = await listDevices(navigator.mediaDevices);
      const cur = stream?.getVideoTracks()[0]?.getSettings().deviceId;
      const next = nextDeviceId(devs.cameras, cur);
      if (next) {
        safeSet('videoDeviceId', next);
        void openWith({ ...choice, videoDeviceId: next });
      }
    });
    torchBtn.addEventListener('click', async () => {
      const t = stream?.getVideoTracks()[0];
      if (!t) return;
      try {
        torchOn = !torchOn;
        await setTorch(t, torchOn);
        torchBtn.textContent = torchOn ? 'Torch on' : 'Torch off';
        torchBtn.setAttribute('aria-pressed', String(torchOn));
      } catch {
        torchOn = !torchOn;
      }
    });

    if (!(await openWith(choice))) return;
    if (!consent.state.camera) return;

    // MediaPipe smoke path (off the critical path: throttled, async init).
    if (g !== gen) return;
    const session = new PresenceSession(onVideoFrame, (yes) => presence.setValue(yes ? 'Person in view: yes' : 'Person in view: no', null));
    presenceSession = session;
    try {
      await session.load(createPoseSmoke, video);
      if (!session.isEnded) presence.setValue('Looking…');
    } catch (err) {
      if (!session.isEnded) presence.setValue('Pose model failed to load: ' + (err instanceof Error ? err.message : String(err)));
    }
  }

  /** Fingertip heart-rate mode: back camera, torch on when supported, ROI frame loop -> core. */
  async function beginFingertip(): Promise<void> {
    const g = ++gen;
    const view = buildFingertipView(qualityMeter);
    const stopBtn = el('button', { type: 'button', class: 'btn', id: 'fp-stop' }, 'Stop');
    stopBtn.addEventListener('click', () => showConsent());
    view.root.append(el('div', { class: 'btnrow' }, stopBtn));
    main.replaceChildren(view.root, notOffered);
    let opened: MediaStream;
    try {
      // Prefer the back camera (a stale remembered deviceId from the standard mode is ignored here).
      const r = await openStreamWithFallback(consent, navigator.mediaDevices, { facing: 'environment' });
      opened = r.stream;
    } catch (err) {
      if (g === gen) showConsent(describeMediaError(err));
      return;
    }
    if (g !== gen) {
      stopStream(opened);
      return;
    }
    stream = opened;
    const vtrack = stream.getVideoTracks()[0];
    if (!vtrack) {
      showConsent('No camera track was available.');
      return;
    }
    let torch = false;
    if (supportsTorch(vtrack)) {
      try {
        torch = await setTorch(vtrack, true);
      } catch {
        torch = false;
      }
    }
    view.setNoTorchNote(!torch);
    view.status.textContent = `${vtrack.label || 'Camera'}${torch ? ', flash on' : ', no flash'}`;
    view.video.srcObject = new MediaStream([vtrack]);
    try {
      await view.video.play();
    } catch {
      /* needs a gesture on some browsers; Start was one */
    }
    if (g !== gen) return;
    const session = new FingertipSession(onVideoFrame, view.hooks);
    fingertipSession = session;
    session.start(view.video);
  }

  function fill(sel: HTMLSelectElement, devices: MediaDeviceInfo[], currentId: string | undefined, fallback: string): void {
    sel.replaceChildren(
      ...devices.map((d, i) => {
        const o = el('option', { value: d.deviceId }, d.label || `${fallback} ${i + 1}`);
        o.selected = d.deviceId === currentId;
        return o;
      }),
    );
  }

  showConsent();
}
