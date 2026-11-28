export const NOT_OFFERED: { name: string; reason: string }[] = [
  { name: 'SpO₂ (blood oxygen)', reason: 'A phone camera and ordinary light cannot measure it reliably without a calibrated medical sensor.' },
  { name: 'Blood pressure', reason: 'Camera pulse signals are not accurate enough to estimate it, and wrong numbers could mislead you.' },
  { name: 'Glucose', reason: 'It cannot be measured from video or sound; any claim to do so would be unfounded.' },
  { name: 'Temperature', reason: 'A normal camera does not measure body temperature; that needs a calibrated thermal sensor.' },
  { name: 'Any disease or diagnosis', reason: 'CareLens is not a medical device and cannot tell whether you are well or ill. Talk to a clinician.' },
  { name: 'Emotion or mental-health scores', reason: 'Facial and voice cues do not reliably reveal feelings, and scoring them can be harmful and biased.' },
  { name: 'Stroke or facial-asymmetry screening', reason: 'This is a medical emergency judgement; if you suspect a stroke, call emergency services immediately.' },
  { name: 'Age or identity', reason: 'Privacy by design: CareLens never identifies, recognises or profiles people.' },
];
