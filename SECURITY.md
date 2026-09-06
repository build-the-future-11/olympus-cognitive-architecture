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
to public addresses. Local files are byte-bounded and can be restricted to
explicit read roots; filesystem writes are denied by default. Mutating API
operations are loopback-only unless `OLYMPUS_API_TOKEN` is configured, in which
case clients must send `Authorization: Bearer <token>`. Do not expose Olympus
through a reverse proxy without configuring that token. Run logs, datasets, and external credentials must be
reviewed before sharing. The alpha release is not a certification for
regulated, safety-critical, or multi-tenant production use.
