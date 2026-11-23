# AI development workflow — coordinator, subagents, skills, tokens

How to build CareLens in Claude Code like a coordinated team. The main session is the
**coordinator**; it dispatches specialized **subagents** (the Agent/Task tool) and uses
**skills**, keeping its own context small and its token spend low.

## Roles

**Coordinator (main session).** Owns `docs/07-BUILD-PLAN.md` and the task list. Decides the
next task, dispatches subagents, reviews their results, keeps `pytest -q` green, and commits.
It does *not* do bulk searching or file-reading itself — that's what burns context.

**Subagents (Agent/Task tool).** Each runs in its own context and returns a *conclusion*, not
raw logs:
- **Explore** — read-only code/file search ("where is the fall threshold set?", "all
  `VideoSource` callers?"). Keeps big file dumps out of the coordinator's context.
- **Plan** — design a task before coding (SQLite schema, camera-registry shape, TensorRT swap).
- **general-purpose** — multi-step implementation needing many tool calls.
- **fork** — a parallel line of work that needs the current context.
- Use `isolation: worktree` for parallel edits so two agents don't collide on the same files.

## Skills → task map (from the available catalog)

- **engineering:system-design / architecture** — new components (camera registry, store, Jetson swap).
- **engineering:testing-strategy** — per-module test plans; keep the synthetic-ground-truth approach.
- **engineering:code-review** — review every diff before commit (edge cases, security, perf).
- **engineering:debug** — structured debugging when a test fails or a camera path misbehaves.
- **engineering:documentation** — READMEs, runbooks, `LIMITATIONS.md`.
- **dataviz** — dashboard charts (HR/BR trends, timeline) so they read as one system.
- **data:statistical-analysis / validate-data** — the accuracy report (HR error vs Apple Watch, stats).
- **bio-research:** (methodology) — sanity-check the rPPG/breathing science and the skin-tone validation design.
- **anthropic-skills:deep-research** — only for real research questions (rPPG methods, ONVIF, Jetson
  TensorRT pose, skin-tone bias literature); it spawns its own subagents, so use it deliberately.

## Token-efficient orchestration

- **One build-plan task at a time.** Coordinator dispatches; it doesn't read in bulk.
- **Route searches to Explore** so file dumps never enter the main context.
- **Parallelize independent subagents** (e.g. Explore + Plan at once), but cap concurrency to
  stay under budget.
- **Model per role:** coordinator on **Sonnet**; Explore/routine on **Haiku**; hard
  design/debug/signal-math on **Opus** (see `CLAUDE.md`).
- **Subagents return conclusions, not logs.** Coordinator owns the task list and commits.
- **Budget:** stop long runs at **80% of the weekly limit**; **never** extra-usage credits.

## Phase → who does what (concrete)

| Phase (see 07-BUILD-PLAN) | Plan agent | Explore agent | Implement | Review |
|---|---|---|---|---|
| P1 registry + RTSP/ONVIF + SQLite | schema + config.yaml + DB schema | map `VideoSource` touchpoints | general-purpose + tests | code-review skill |
| P2 multi-camera + alerting | worker/supervisor design | find dashboard/event seams | general-purpose + tests | code-review |
| P3 Jetson (TensorRT/GStreamer) | perception-swap design | find pose provider seams | general-purpose + perf | code-review |
| P4 validate + present | accuracy-analysis plan | — | data/stats skills | human sign-off |

## Guardrails every agent inherits (from CLAUDE.md)

Wellness wording only; consented or synthetic footage only; **cloud-app cameras = clip-export only via the vendor app**; permissive licenses only; no secrets committed; local-first. Keep
`pytest -q` green; small diffs; one commit per task.
