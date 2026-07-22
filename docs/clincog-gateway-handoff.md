# ClinCog classification gateway handoff

ICVision posts one bounded `image/webp` body to the fixed ClinCog endpoint
`/v1/ic-classifications` using only its inbound `CLINCOG_API_TOKEN`.

The server, not ICVision, must inject `OPENAI_API_KEY` from 1Password, call
`api.openai.com/v1/responses` with the fixed reviewed model, prompt, schema,
and `store:false`, never forward the ClinCog token, and return only
`label`, `confidence`, `reason`, `model`, and `request_id`. Deployment and
gateway administration are outside this repository.
