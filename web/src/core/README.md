DOM-free CareLens core, ported from Python.

Nothing in this folder may touch the DOM, `window`, or MediaStream APIs, so it runs in a
Web Worker and under Vitest in Node. Thresholds come from `shared/params.json`.
