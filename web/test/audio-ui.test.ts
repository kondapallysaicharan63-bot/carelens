import { describe, expect, it } from 'vitest';
import { BREATH_SOUND_HINT, COUGH_NOTE, breathSoundGuidance, coughLabel } from '../src/ui/audioCards';

describe('audio card wording', () => {
  it('uses wellbeing wording and the agreed guidance', () => {
    expect(BREATH_SOUND_HINT).toContain('10–20 cm');
    expect(BREATH_SOUND_HINT).toContain('don’t talk');
    expect(COUGH_NOTE).toBe('Counts sounds the model recognises as coughs — wellbeing trend only.');
    expect(coughLabel(3)).toBe('Cough sounds this session: 3');
    const all = (['off', 'idle', 'no_mic', 'too_quiet', 'clipping', 'talking', 'settling', 'low_signal', 'ambiguous', 'ok'] as const).map((r) => breathSoundGuidance(r));
    all.push(BREATH_SOUND_HINT, COUGH_NOTE);
    for (const t of all) expect(t).not.toMatch(/emergenc|diagnos|apnea|pneumonia|covid|asthma|disease/i);
  });
  it('every gate reason has its own guidance', () => {
    expect(new Set(['no_mic', 'too_quiet', 'clipping', 'talking', 'settling', 'low_signal', 'ambiguous', 'ok'].map((r) => breathSoundGuidance(r as never))).size).toBe(8);
    expect(breathSoundGuidance('ambiguous')).toBe('Breathing sounds are ambiguous — breathe a little more audibly on the out-breath, or use the camera breathing card.');
  });
});
