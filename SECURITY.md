# Security Policy

## Supported versions

Security fixes are applied to the current `0.2.x` alpha line.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, private
data, or unredacted logs. Once the repository is published on GitHub, use the
repository's **Security → Report a vulnerability** form so the report remains
private. Include the affected version, reproduction steps, impact, and the
smallest safe proof of concept.

Until a private reporting channel is configured by the repository owner, do
not transmit vulnerability details. The absence of that owner-controlled
channel is a publication gate recorded in `RELEASE.md`.

## Operating boundary

Olympus defaults network ingestion to denied. HTTP ingestion requires an
explicit domain allowlist and accepts only credential-free HTTPS URLs resolving
to public addresses. Run logs, datasets, and external credentials must be
reviewed before sharing. The alpha release is not a certification for
regulated, safety-critical, or multi-tenant production use.

The API is designed for loopback-only development by default. Before binding it
to a non-loopback interface, set a strong `OLYMPUS_API_TOKEN`; stateful and
compute-bearing routes then require `Authorization: Bearer <token>`. Set
`OLYMPUS_MAX_CONCURRENT_WORKLOADS` to bound simultaneous work. These controls do
not make the alpha service suitable for untrusted multi-tenant deployment.
