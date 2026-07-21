# OpenAI modernization — Gate 1 offline request and result contract

## Status

Gate 1 is an offline-only semantic contract. It has no core or CLI wiring, no
credential lookup, no network behavior, and no authorization for live use.

## Reviewed synthetic contract

- One immutable fixture owns the fixed model alias, `low` image detail, request
  limits, response limit, output-token bound, timeout value, and concurrency of
  one. Timeout and concurrency enforcement belongs to Gate 2 transport work.
- The builder accepts only the exact module-owned synthetic prompt and WebP
  fixture. It returns immutable final UTF-8 JSON bytes plus safe metadata.
- The Responses body contains one user item with ordered `input_text` then
  `input_image` content, a `data:image/webp;base64` URL, `store:false`, bounded
  `max_output_tokens`, and strict `text.format` JSON Schema.
- Temperature, retries, reasoning, tools, state, streaming, and background
  fields are deliberately absent.
- Completed results accept only a bounded parsed classification mapping. Usage is
  reduced to known bounded token counts, including nested cached tokens. Raw
  response material, headers, exceptions, and provider reason text are not kept.

## SDK characterization and intentional deltas

Recorded offline evidence: 76 synthetic tests include a locally intercepted
current-SDK call. The SDK serializes two ordered user inputs and `temperature`
of `0.2`. Gate 1 compares model, prompt, MIME, image bytes, and input order as
semantic evidence; it does not claim wire parity.

Intentional raw-contract deltas are a combined typed content array, `store:false`,
fixed bounds, strict schema, explicit `low` detail, and omitted temperature.
Reasoning remains omitted until capability verification. These are offline design
constraints, not claims of gateway or model support.

## Gate 2 deferrals

Gate 2 must separately review and enforce transport construction, credential
injection, timeout and concurrency behavior, redirects, retries, status handling,
MIME handling, response-size measurement, and gateway capability checks.

Live or protected-data use remains blocked pending redacted retention, logging,
origin, TLS, model, store, and schema evidence plus institutional approval.
Cloudflare, Kamal, and CLIProxy deployment are outside ICVision scope.
