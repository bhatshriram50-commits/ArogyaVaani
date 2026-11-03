# API reference

Protected central endpoints require `Authorization: Bearer <JWT>`. `task` is PNEUMONIA, RETINOPATHY, or BRAIN_TUMOR.

| Endpoint | Access | Purpose |
| --- | --- | --- |
| `GET /api/health`, `GET /api/ready` | public | Liveness and readiness without secrets |
| `POST /api/auth/register`, `/login`, `/google` | public | Account creation/sign-in; rate-limited |
| `GET /api/auth/session`, `POST /api/auth/logout` | signed user | Session validation/logout |
| `GET /api/medical-tasks` | signed user | Enabled task metadata |
| `POST/GET /api/federated/rounds` | admin | Create/list task rounds |
| `POST /api/federated/rounds/:round/updates?samples=N&task=T` | selected hospital | AVM1 binary-only upload |
| `GET /api/federated/rounds/active`, `/status` | signed user | Round/task status |
| `POST /api/federated/rounds/:round/cancel` | admin | Cancel incomplete round |
| `GET /api/federated/global-model[-binary]` | selected hospital | Authorized active task model/checksum |
| `POST/GET /api/training/runs` | hospital | Caller-owned local-training metadata |
| `POST /api/predictions/metadata` | hospital | Non-image prediction metadata |
| `GET /api/admin/*` | admin | Dashboard, hospitals, rounds, models, updates, audit logs |
| `POST /api/admin/models/:version/rollback?task=T` | admin | Activate historical task model without deletion |

Common errors: 400 validation, 401 JWT, 403 authorization, 409 state/duplicate conflict, 413 size limit, and 503 unavailable persistence/artifacts.

The hospital-local service provides `/health`, `/ready`, `/medical-tasks`, `/dataset/upload`, `/dataset/validate`, `/train`, `/training/status[/<job_id>]`, `/predict`, `/federated/submit`, and `/federated/download-global`. It is hospital-local only. `POST /train` returns `{ job_id, status: "queued" }`; poll until `completed` or `failed`.
