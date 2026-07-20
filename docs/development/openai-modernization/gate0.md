# OpenAI modernization ? Gate 0 ACTIVE record

## Grounding

Parent goal: create a safe, reversible boundary before any raw OpenAI Responses
transport is enabled. `src/icvision/cli.py` currently parses API options and starts
logging before calling `core.label_components`; `src/icvision/core.py` currently
validates keys and loads EEG/ICA inputs before classifying through the SDK.

## Invariants

- SDK remains the default and preserves legacy `base_url` and `--api-key`.
- Raw requires exactly `openai-responses`; it cannot use a custom endpoint.
- Validation is pure and precedes credential resolution, logging, data reads,
  output creation, plotting, clients, sockets, and provider calls.
- Raw is deliberately non-executable in Gate 0.
- Raw custom CA overrides fail closed; proxy and endpoint ambient variables do
  not select or change the reviewed profile.

## Highest-learning question

Can one small pure policy seam block unsafe raw selections before every existing
CLI/core pipeline side effect while leaving SDK behavior compatible?

## Provisional ladder and climb semantics

1. **Gate 0 ? policy seam:** immutable reviewed profile, early validation, raw stop.
2. **Gate 1 ? private transport:** fixed-host HTTPS request construction only.
3. **Gate 2 ? response adaptation:** map successful and incomplete Responses data.
4. **Gate 3 ? retirement:** remove SDK call path only after compatibility evidence.

Only Gate 0 is lit. A rung is complete only with focused offline evidence and an
independent review boundary; later rungs remain unapproved.

## NEXT self-review

Implement only the profile/policy seam, append optional core parameters, add CLI
arguments, and add synthetic offline tests. Confirm no endpoint override, proxy,
credential value, socket, data, plot, output, dependency, retry, or fallback
surface has been introduced. Stop if preserving legacy SDK behavior requires
changing public defaults or endpoint policy.

## Intermediate specification

The independently useful result is a callable pure validator returning either an
immutable SDK/raw selection or a stable sanitized error. Its consumer proof is:

`python -B -m pytest tests/test_transport_policy.py -q --maxfail=1 --no-cov -p no:cacheprovider`

## Plan-quality check

This is the smallest change that tests the decision boundary. It avoids a
transport abstraction, new dependency, custom endpoint, raw request, fallback,
or scientific behavior change. Rollback is one isolated Gate 0 commit revert.

## Exact evidence and step-back pending

Base: `ea515fbbe1fbba1b326f2f3f9d91320a369449c5`.
Branch: `codex/openai-http-modernization`.
Pending: run the focused node, then Gate 0, compatibility, and cumulative Make
targets. Step back to Raven if any test shows SDK behavior changed, policy
validation occurs after side effects, or the reviewed-patch fallback fails.

Compatibility note: `tests/test_core.py::test_label_components_custom_params`
fails identically at unchanged base `ea515fb`: the test expects
`icvision_results.csv`, while current code writes a subject-prefixed results
filename. The recurring compatibility targets temporarily deselect only this
known base-failing node; repairing that test contract is separate work.


## Learned and step-back

- Small enough: yes; the increment is one policy seam and early consumers.
- Technical proof: 22 focused offline tests pass; the cumulative compatibility run
  previously passed 56 tests with one proven base-failing node deselected.
- Consumer proof: `make test-openai-gate0` executes with the Windows-resolved
  `OPENAI_PYTHON=python` without changing legacy Make targets.
- Beneficiary: CLI and Python callers now receive sanitized fail-before-side-effect
  raw-lane decisions while the SDK default remains compatible.
- Direction: invariants remain open and valid; raw execution is still disabled.
- Wrong-path signals: a direct raw API key could bypass the CLI-only rule, and
  privileged patch tooling was mixed into the transport commit. Both were caught
  by independent review: the API gap was closed and tooling was separated.
- Migration learning: policy must be enforced at the shared Python boundary, not
  only at the CLI. Privileged delivery tooling is not part of transport policy.
- Retirement progress: closer by one independently tested policy boundary; no SDK
  call path has been removed.
- Verdict: `CONTINUE` for Gate 0 only. Later rungs remain provisional and
  unapproved; Gate 1 must start from this evidence and preserve these invariants.
