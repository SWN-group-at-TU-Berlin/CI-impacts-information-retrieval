# Project Spec — CI Impacts Information Retrieval

## Planning

* **Problem / project description**: I want others to access my generated data, which is stored in a Postgres database.
* **Goals & outcomes**: Users can access my data through an API. Only certain users may access the entire data; others have only partial access. The database itself is never visible outside the project.
* **Stakeholders & users**: I am the admin. There are 2 user groups: group A with full access to the database, group B with only partial access (via the API).
* **User needs / use cases**: Users send requests to an API to access the data stored in the database.
* **Scope**: Three Docker containers:
  1. **Extraction container** — runs the LLM extraction workflow that extracts structured data from documents; responses are postprocessed and the final output is stored in the database. It updates the database regularly with new information.
  2. **Database container** — Postgres; stores the extracted data; only reachable from inside the project network.
  3. **API container** — serves the data to users and provides user management (group A = full access, group B = partial access).
* **Requirements & constraints**: The database must not be visible outside the project — users can only reach it through the API.
* **Development framework**: Use an agile for this software and include me in feedback loops often.
* **Feasibility**: Project is public.
* **Resources & responsibilities**: The agent does the coding; I am frequently included to provide feedback (agile approach).

## Technical Stack

* **Frontend**: none — the product is API-only. FastAPI's auto-generated OpenAPI/Swagger UI (`/docs`) serves as the interface for browsing and manual testing. A React/Vite frontend can be added later only if real users need data browsing.
* **Backend**: FastAPI (Python 3.12 — same language and toolchain as the existing extraction pipeline, so Pydantic models are shared between extraction output, DB schema, and API responses).
  * SQLAlchemy 2.0 (asyncpg driver) for DB access, Alembic for migrations, Pydantic v2 + pydantic-settings for schemas and config.
* **UI/UX**: none.
* **Testing**: Pytest only — unit tests plus integration tests against a real Postgres; no end-to-end tests.
* **Build**: create 3 Docker containers orchestrated with Docker Compose.

## Architecture & Data Flow

1. `extraction` container → writes postprocessed extraction output to Postgres via a dedicated, insert-only DB role.
2. `database` container (Postgres) → attached to an internal Docker network only; **no published port** (5432 must never be exposed to the host or outside).
3. `api` container (FastAPI) → the only container that publishes a port; authenticates users, enforces access groups, reads from Postgres.

## Access Control Model

* **Authentication**: login with email/password issues short-lived JWTs (JSON Web Tokens) whose role claim (`group_a` / `group_b`) travels with every request.
* **Two-layer authorization** (defense in depth):
  1. API layer — FastAPI dependencies gate endpoints and filter response fields per group (this also covers "partial" = column-level restrictions).
  2. Postgres layer — Row-Level Security (RLS): the API sets the user group per request (`SET LOCAL`), and RLS policies enforce row limits even if the API logic contains a bug.
* Optional: API keys for machine-to-machine clients.

## Security Constraints (enforced from the first build step)

* No published database port; DB and extraction containers on an internal Compose network.
* Secrets via `.env` / Compose secrets — no hard-coded passwords (the current `POSTGRES_PASSWORD: postgres` in `compose.yaml` must be replaced).
* Passwords hashed with argon2/bcrypt.
* Least-privilege DB roles: one role per service (extraction = insert-only, api = read per policy).

## Delivery Pipeline (agile, contract-first)

1. **Contract**: define table schemas, endpoint list, and the role/permission matrix: "partial" for group B means that the should have access to only certain rows and columns in the database
2. **Scaffold**: Compose file with 3 services + internal network (no DB port), FastAPI skeleton, Alembic, pytest setup.
3. **Vertical slices** (each independent, pytest-verified, followed by a feedback round):
   * migrations + seed data
   * auth (register/login, JWT, roles, admin account)
   * one read endpoint with group A/B behavior + RLS
   * remaining endpoints (+ API keys)
   * ingestion contract for the extraction container
4. **Verification per slice**: pytest unit + integration tests (real Postgres, including tests that connect as each group role and assert what is visible); ruff + type checks in CI.

## FAQ

* **Q: Which columns should be stored in the database?**
  A: Use the CSV file `/home/a-buch/Documents/TUB_DWN/_PROJECTS/interim_results_debug/llm_geollm_chain_step1_Koks 2022.csv` as the example for the columns that should be stored in the database.

* **Q: Which data is hidden from users?**
  A: The column with `chunk_text` should be hidden from the user.

* **Q: Can users register themselves?**
  A: No, users should not register themselves.

