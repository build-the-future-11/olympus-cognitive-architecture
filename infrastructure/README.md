# Infrastructure

Local development relies on the root `docker-compose.yml` file, which provisions:

- PostgreSQL with `pgvector`,
- Redis for cache and queue compatibility.

The Python implementation keeps local SQLite and in-memory fallbacks so the core demos work without containers.

