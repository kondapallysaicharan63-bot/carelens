import { detectBlackVideo } from '../capture/blackframe';
import { describeMediaError, listDevices, nextDeviceId, setTorch, supportsTorch, type CaptureChoice, type Facing } from '../capture/devices';
import { onVideoFrame } from '../capture/frames';
import { PoseSession } from '../capture/poseSession';
import { createPoseLandmarker } from '../capture/pose';
import { createHandLandmarker } from '../capture/hand';
import { PauseGroup } from '../capture/pauseGroup';
import { VideoChestSource } from '../capture/chestSampler';
import params from '../../../shared/params.json';
import { createFaceLandmarker } from '../capture/face';
import { openStreamWithFallback, stopStream } from '../capture/session';
import { safeGet, safeRemove, safeSet } from '../storage';
import { BLINK_TITLE, makeCard, TIER1, TIER2, type Card } from './cards';
import { buildFingertipView, FingertipSession, HRV_PARAMS } from './fingertip';
import { HRV_NOTE, HRV_OFF_TEXT } from './hrvCard';
import { HrvRecorder } from '../export/hrv';
import { buildFaceCard, FaceHrSession, type FaceCard } from './faceHr';
import { qualityMeter } from './quality';
import { buildPoseCards, type PoseCards } from './poseCards';
import { buildAudioCards, buildAudioOffCards, type AudioCards } from './audioCards';
import { AudioSession } from '../capture/audioSession';
import { createYamnet } from '../capture/yamnet';
import { Consent, DISCLAIMER } from './consent';
import { el } from './dom';
import { NOT_OFFERED } from './notOffered';
import { buildExportPanel } from './exportPanel';
import { SessionRecorder } from '../export/recorder';
import type { Mode as RecMode } from '../export/schema';
import { BreathCameraLogger } from '../export/breathCamera';
import { CoughRecorder } from '../export/cough';
import { BLINK_NOTE, buildBlinkCard, type BlinkCard } from './blinkCard';
import { ACTIVITY_NOTE, buildActivityCard, type ActivityCard } from './activityCard';
import { buildTremorTest, TREMOR_NOTE, TREMOR_PARAMS, type TremorTest } from './tremorTest';
import { buildMovementTests, MOVEMENT_PARAMS, MOVEMENT_NOTE, type MovementTests } from './movementTests';

type Theme = 'auto' | 'light' | 'dark';
type Mode = 'standard' | 'fingertip';

export function mountApp(root: HTMLElement): void {
  const consent = new Consent();
  let stream: MediaStream | null = null;
  let choice: CaptureChoice = {};
  let poseSession: PoseSession | null = null;
  let faceSession: FaceHrSession | null = null;
  /** pose and face sessions (the blink card shares the face one) are held while the hand test runs, also ones created after it started */
  const handTestHold = new PauseGroup();
  let fingertipSession: FingertipSession | null = null;
  let audioSession: AudioSession | null = null;
  let audioCards: AudioCards | null = null;
  let movementTests: MovementTests | null = null;
  let tremorTest: TremorTest | null = null;
  let mode: Mode = 'standard';
  // The user's standard-mode toggles, kept while fingertip mode forces camera-only.
  let stdCamera = consent.state.camera;
  let stdMic = consent.state.mic;
  const dataPanel = buildExportPanel();
  let recorder: SessionRecorder | null = null;
  const startRecorder = (modes: RecMode[]): SessionRecorder => {
    const r = new SessionRecorder(modes);
    recorder = r;
    dataPanel.setRecorder(r);
    return r;
  };
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
      dataPanel.root,
      notOffered,
    );
  }

  function resultsSections(pose?: PoseCards, face?: FaceCard, audio?: AudioCards | null, movement?: MovementTests, tremor?: TremorTest, blink?: BlinkCard, activity?: ActivityCard): HTMLElement {
    const wrap = el('div', {});
    const r = el('section', { class: 'results', 'aria-labelledby': 'h-results' }, el('h2', { id: 'h-results' }, 'Results'), el('p', { class: 'sub' }, 'Tier 1: wellbeing trends with a signal-quality meter.'));
    const grid = el('div', { class: 'grid' });
    grid.append(pose ? pose.inView : makeCard('In view', 1, 'Starts when the camera is on.').root);
    grid.append(pose ? pose.posture : makeCard('Posture', 1, 'Standing, sitting or lying. Starts when the camera is on.').root);
    for (const [t, n] of TIER1) {
      if (t === 'Heart rate (face)' && face) grid.append(face.root);
      else if (t === 'Breathing rate' && pose) grid.append(pose.breathing);
      else if (t === 'Fall and inactivity check-in' && pose) grid.append(pose.checkins);
      else if (t === 'Breathing (sound)') grid.append((audio ?? buildAudioOffCards()).breathing);
      else if (t === 'Cough sounds') grid.append((audio ?? buildAudioOffCards()).cough);
      else grid.append(makeCard(t, 1, n).root);
    }
    r.append(grid);
    const e = el('section', { class: 'results experimental', 'aria-labelledby': 'h-exp' }, el('h2', { id: 'h-exp' }, 'Experimental'), el('p', { class: 'sub' }, 'Tier 2: early, less reliable features. Treat as curiosity, not information.'));
    const g2 = el('div', { class: 'grid' });
    for (const [t, n] of TIER2) {
      if (t === 'Heart-rate variability') g2.append(makeCard(t, 2, HRV_NOTE, HRV_OFF_TEXT).root);
      else if (t === 'Sit-to-stand and walking speed') g2.append(movement ? movement.root : makeCard('Movement tests', 2, MOVEMENT_NOTE, 'Starts when the camera is on.').root);
      else if (t === 'Hand movement') g2.append(tremor ? tremor.root : makeCard('Hand movement', 2, TREMOR_NOTE, 'Starts when the camera is on.').root);
      else if (t === BLINK_TITLE) g2.append(blink ? blink.root : makeCard(BLINK_TITLE, 2, BLINK_NOTE, 'Starts when the camera is on.').root);
      else if (t === 'Activity') g2.append(activity ? activity.root : makeCard('Activity', 2, ACTIVITY_NOTE, 'Starts when the camera is on.').root);
      else g2.append(makeCard(t, 2, n).root);
    }
    e.append(g2);
    wrap.append(r, e);
    return wrap;
  }

  // ---- live ----
  function stopAll(): void {
    gen++;
    if (poseSession) handTestHold.remove(poseSession);
    poseSession?.end();
    poseSession = null;
    if (faceSession) handTestHold.remove(faceSession);
    faceSession?.end();
    faceSession = null;
    fingertipSession?.end();
    fingertipSession = null;
    audioSession?.end();
    audioSession = null;
    audioCards = null;
    movementTests?.disable('The camera is off, so movement tests cannot run. Start the camera to try one.');
    movementTests = null;
    tremorTest?.disable('The camera is off, so the hand test cannot run. Start the camera to try it.');
    tremorTest = null;
    if (recorder) void dataPanel.sessionEnded(recorder);
    stopStream(stream);
    stream = null;
  }

  async function beginCapture(): Promise<void> {
    const g = ++gen;
    const rec = startRecorder([...(consent.state.camera ? ['camera' as const] : []), ...(consent.state.mic ? ['microphone' as const] : [])]);
    const pose = consent.state.camera ? buildPoseCards(qualityMeter) : undefined;
    const activity = pose ? buildActivityCard({ record: (row) => void rec.upsertSummary(row) }) : undefined;
    const movement = pose ? buildMovementTests({ mp: MOVEMENT_PARAMS, record: (row) => void rec.addMobilitySummary(row), heightSaving: () => dataPanel.savingEnabled() }) : undefined;
    movementTests = movement ?? null;
    if (pose) {
      const h = pose.hooks;
      const brLog = new BreathCameraLogger(rec, params.vitals.breathing.step_s);
      pose.hooks = {
        ...h,
        onView(v) {
          h.onView(v);
          brLog.onView(v);
          movement?.onView(v);
          activity?.onView(v);
        },
        onEvent(e) {
          h.onEvent(e);
          rec.addEvent({ kind: e.kind === 'fall' ? 'possible_fall' : 'inactivity', agoS: e.tConfirmed - e.tOnset, trigger: e.details.trigger });
        },
      };
    }
    const video = el('video', { playsinline: '', muted: '', autoplay: '', 'aria-label': 'Camera preview' });
    video.muted = true;
    video.playsInline = true;
    // Hand test (W8): the hand model loads only when the test starts and is closed afterwards. End-to-end builds (vite --mode e2e)
    // replay scripted hand landmarks instead; the condition is a build-time constant, so that provider is removed from production builds.
    const tremor = consent.state.camera
      ? buildTremorTest({
        onActive: (active) => { handTestHold.hold(active); if (active) movement?.reset(); },
        getDeclaredFps: () => stream?.getVideoTracks()[0]?.getSettings?.().frameRate ?? null,
        tp: TREMOR_PARAMS, record: (row) => void rec.addMobilitySummary(row), subscribe: onVideoFrame, getVideo: () => video,
        loadHand: async () => (import.meta.env.VITE_E2E === '1' ? (await import('../e2e/scriptedHand')).createScriptedHand : createHandLandmarker)(),
      })
      : undefined;
    tremorTest = tremor ?? null;
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
    const face = consent.state.camera ? buildFaceCard(qualityMeter) : undefined;
    if (face) {
      const h = face.hooks;
      face.hooks = {
        ...h,
        onUpdate(u) {
          h.onUpdate(u);
          rec.addEstimate({ metric: 'hr_face', value: u.bpm, quality: u.quality, reported: u.reported && u.bpm !== null, reason: u.reason, span_s: u.span_s });
        },
      };
    }
    const blink = consent.state.camera
      ? buildBlinkCard({ meter: qualityMeter, record: (row) => void rec.upsertSummary(row), getDeclaredFps: () => stream?.getVideoTracks()[0]?.getSettings?.().frameRate ?? null })
      : undefined;
    if (face && blink) {
      const h = face.hooks;
      face.hooks = { ...h, onLandmarks: (f) => blink.onFrame(f) };
    }
    const audio = consent.state.mic ? buildAudioCards(qualityMeter) : null;
    if (audio) {
      const h = audio.hooks;
      const coughs = new CoughRecorder(rec);
      audio.hooks = {
        ...h,
        onBreath(u) {
          h.onBreath(u);
          rec.addEstimate({ metric: 'br_sound', value: u.bpm, quality: u.quality, reported: u.reported && u.bpm !== null, reason: u.reason, span_s: u.span_s });
        },
        onCough(r) {
          h.onCough(r);
          coughs.update(r); // stored when first seen, updated while the tracker extends or merges it
        },
      };
    }
    audioCards = audio;
    if (pose) live.append(pose.banner);
    if (consent.state.camera) live.append(el('div', { class: 'preview roi-host' }, video, face!.overlay));
    live.append(controls);
    main.replaceChildren(live, resultsSections(pose, face, audio, movement, tremor, blink, activity), dataPanel.root, notOffered);

    let torchOn = false;
    const openWith = async (c: CaptureChoice): Promise<boolean> => {
      audioSession?.end(); // release the old microphone graph before its stream stops
      audioSession = null;
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
      poseSession?.restart(); // re-bind the frame loops to the (new) stream
      faceSession?.restart();
      blink?.reset(); // the eye baseline and the one-minute window start over with the new camera
      activity?.breakStream();
      if (audioCards && stream) startAudio(g, audioCards, stream); // (re)start the audio chain; the cough count starts again
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
    if (g !== gen) return;
    if (!consent.state.camera) return;

    // Face heart rate: FaceLandmarker on every frame, but only while its card is active (front camera is the default).
    if (g !== gen) return;
    const fsession = new FaceHrSession(onVideoFrame, face!.hooks);
    faceSession = fsession;
    handTestHold.add(fsession); // the hand test can start before the camera has opened
    face!.onActiveChange((a) => fsession.setActive(a));
    blink?.onActiveChange((a) => fsession.setBlinkActive(a)); // one face model for both cards: it runs while either is on
    // End-to-end builds only (vite --mode e2e): replay scripted eye landmarks when the test page supplies them, else the real model.
    const loadFace = import.meta.env.VITE_E2E === '1' ? async () => (await import('../e2e/scriptedFace')).createScriptedFace(createFaceLandmarker) : createFaceLandmarker;
    const faceLoad = fsession.load(loadFace, video).then(
      () => { if (!fsession.isEnded) { face!.setLoadState('ready'); blink?.setLoadState('ready'); } },
      (err) => {
        if (fsession.isEnded) return;
        const m = err instanceof Error ? err.message : String(err);
        face!.setLoadState('failed', m);
        blink?.setLoadState('failed', m);
      },
    );

    // Pose: presence, posture, fall + inactivity check-ins and breathing, scheduled at ~8 Hz (PoseScheduler).
    const session = new PoseSession(onVideoFrame, pose!.hooks, (v) => new VideoChestSource(v, params.chest_shift));
    poseSession = session;
    handTestHold.add(session);
    pose!.onInactivityChange((sec) => session.setInactivitySeconds(sec));
    try {
      // End-to-end builds only (vite --mode e2e): replay a scripted landmark sequence instead of the real model.
      // The condition is a build-time constant, so the dynamic import and the provider are removed from production builds.
      const loader = import.meta.env.VITE_E2E === '1' ? (await import('../e2e/scriptedPose')).createScriptedPose : createPoseLandmarker;
      await session.load(loader, video);
      if (!session.isEnded) pose!.setStatus('Looking for you…');
    } catch (err) {
      if (!session.isEnded) {
        pose!.setStatus('Pose model failed to load: ' + (err instanceof Error ? err.message : String(err)));
        movement?.disable('The pose model failed to load, so movement tests cannot run.');
      }
    }
    await faceLoad;
  }

  /** Mic consent is on: worklet capture + envelope + YAMNet. A failure only affects the two audio cards. */
  function startAudio(g: number, cards: AudioCards, s: MediaStream): void {
    audioSession?.end();
    cards.reset();
    const session = new AudioSession(cards.hooks);
    audioSession = session;
    session.start(s).then(
      (running) => { if (running && g === gen && !session.isEnded) void session.loadYamnet(createYamnet); },
      (err) => {
        if (session.isEnded) return;
        cards.reset(); // no stale reading or "no microphone" state from before
        cards.hooks.onYamnet('failed', 'microphone processing unavailable: ' + (err instanceof Error ? err.message : String(err)));
      },
    );
  }

  /** Fingertip heart-rate mode: back camera, torch on when supported, ROI frame loop -> core. */
  async function beginFingertip(): Promise<void> {
    const g = ++gen;
    const rec = startRecorder(['fingertip']);
    const view = buildFingertipView(qualityMeter);
    const vh = view.hooks;
    const hrvRec = new HrvRecorder(rec, HRV_PARAMS.min_duration_s);
    view.hooks = {
      ...vh,
      onHrv(res, segS, epoch, last) {
        vh.onHrv?.(res, segS, epoch, last);
        if (res) hrvRec.update(epoch, res);
      },
      onUpdate(u, m, settleLeft) {
        vh.onUpdate(u, m, settleLeft);
        rec.addEstimate({ metric: 'hr_fingertip', value: u.bpm, quality: u.quality, reported: u.reported && u.bpm !== null, reason: u.reason, span_s: u.span_s });
      },
    };
    const stopBtn = el('button', { type: 'button', class: 'btn', id: 'fp-stop' }, 'Stop');
    stopBtn.addEventListener('click', () => showConsent());
    view.root.append(el('div', { class: 'btnrow' }, stopBtn));
    main.replaceChildren(view.root, view.hrvRoot, dataPanel.root, notOffered);
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
