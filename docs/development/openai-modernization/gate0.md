# OpenAI modernization — Gate 0 fixed gateway policy

## Status

Gate 0 establishes a pure, offline trust boundary for the future raw Responses
lane. It is not authorization for live inference, data handling, or gateway use.

## Fixed ClinCog gateway policy

- The reviewed raw profile fixes HTTPS to `ai.clincognition.com`, port `443`, and
  `/v1/responses`; arbitrary endpoints and URL-shaped profile values are rejected.
- Raw selection rejects a supplied `base_url`, unknown or counterfeit profiles,
  and unsupported transports before credential resolution, data, plots, clients,
  sockets, or provider calls.
- The raw lane uses only system-verified TLS. Ambient endpoint, proxy, and
  provider-key variables do not select or alter its destination; custom CA
  variables fail closed.
- A raw credential may only be supplied through the narrowly named injected
  gateway-credential source. No credential value is read, logged, or stored here.

## SDK compatibility and execution boundary

- The legacy SDK lane remains the default. Its existing custom `base_url`
  compatibility is untouched.
- Raw remains disabled in `core` and `cli`; this policy module does not create a
  client, resolve a credential, send HTTP, or fall through to the SDK.

## Evidence and limits

`python -B -m pytest tests/test_transport_policy.py -q --maxfail=1 --no-cov -p no:cacheprovider`

This gate verifies sanitized fail-before-side-effect policy behavior only. It
does not verify gateway health, model inventory, retention, logging, origin,
certificate operation, or live structured-output support.

## Live-use blocker

Live or protected-data use remains blocked pending redacted gateway operational
evidence, institutional approval, and a separately reviewed transport gate.
Cloudflare, Kamal, and CLIProxy deployment are outside ICVision scope.
