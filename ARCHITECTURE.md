# Architecture

```text
Hospital browser ── HTTPS ── Central Express API ── MongoDB metadata
       │                         │
       │ private network          └── durable AVM1 model artifacts
       ▼
Hospital-local FastAPI + PyTorch ── local datasets/checkpoints/images
```

**RAW MEDICAL DATA NEVER LEAVES THE HOSPITAL ENVIRONMENT.** The central API rejects images, datasets, JSON model submissions, and base64 payloads; it handles only AVM1 parameters plus limited federation metadata.

```text
Admin selects hospitals + task
  → ACTIVE task-scoped round freezes participants/base version
  → selected hospitals train locally and submit one AVM1 update
  → expected count reached → one worker claims AGGREGATING
  → weighted FedAvg → immutable version becomes ACTIVE
  → participants verify checksum and synchronize locally
```

PNEUMONIA, RETINOPATHY, and BRAIN_TUMOR are isolated in dataset paths, models, rounds, updates, predictions, and version numbers. AVM1 declares task, architecture, classes, base version, and tensor metadata, all validated before storage/aggregation.

Email/password accounts use bcrypt hashes and expiring JWTs. Admins are environment-provisioned; Google ID tokens are verified server-side. Role and hospital ownership checks restrict training metadata, selected rounds, and model downloads. MongoDB stores metadata; model artifacts stay outside normal documents. Production does not silently use memory persistence.
