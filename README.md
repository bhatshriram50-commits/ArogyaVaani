# ArogyaVaani: Privacy-Preserving Federated Medical-Image Learning

ArogyaVaani is an academic/clinical research platform for privacy-preserving, task-isolated federated learning on medical images (Pneumonia chest X-rays, Diabetic Retinopathy fundus images, and Brain Tumor MRI scans).

**Privacy Principle:** Raw medical images, patient records, and local datasets strictly remain within hospital boundary nodes. Central aggregation handles only authenticated, cryptographically checked AVM1 mathematical weight updates and coordination metadata.

---

## 1. Prerequisites

- **Node.js:** v20.x or v22.x LTS
- **npm:** v10.x or higher
- **Python:** 3.10+ or 3.11+
- **MongoDB:** v7.0+ or v8.0+ (local instance or MongoDB Atlas cluster)
- **Docker & Docker Compose:** (Optional, for containerized multi-hospital deployment)

---

## 2. Installation Commands

Clone the repository and install dependencies across the monorepo:

```bash
# Install root, backend, and frontend dependencies
npm install

# Set up Python virtual environment and install ML dependencies
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r ml/requirements.txt
```

---

## 3. Environment Variable Configuration

Copy the example environment templates:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
cp ml/.env.example ml/.env
```

### Key Variables:

- **Backend (`backend/.env`):**
  - `PORT`: Central API port (e.g. `4000`)
  - `NODE_ENV`: `development` or `production`
  - `MONGODB_URI`: MongoDB connection URI (e.g. `mongodb://127.0.0.1:27017/arogyavaani` or MongoDB Atlas URI)
  - `MONGODB_DB`: Database name (`arogyavaani`)
  - `JWT_SECRET`: Random key (32+ characters)
  - `ADMIN_EMAIL`: Initial provisioned administrator email
  - `ADMIN_PASSWORD`: Administrator password (min 12 chars)
  - `MODEL_ARTIFACT_ROOT`: Persistent storage directory for AVM1 weights (e.g. `./model_artifacts`)
  - `MODEL_UPDATE_MAX_BYTES`: Maximum payload size (`200mb`)
  - `FRONTEND_ORIGIN`: Allowed origin for CORS (e.g. `http://localhost:5173` or HTTPS domain in production)

- **Frontend (`frontend/.env`):**
  - `VITE_API_URL`: URL to central Express backend (`http://localhost:4000`)
  - `VITE_LOCAL_ML_URL`: URL to local hospital ML FastAPI service (`http://127.0.0.1:8000`)
  - `VITE_GOOGLE_CLIENT_ID`: Optional Google OAuth Client ID

- **Hospital ML Service (`ml/.env`):**
  - `LOCAL_DATA_ROOT`: Local directory for uploaded medical datasets (`./local_datasets`)
  - `LOCAL_MODEL_ROOT`: Local directory for model checkpoints (`./local_models`)
  - `LOCAL_UPDATE_ROOT`: Local directory for pending updates (`./local_updates`)
  - `LOCAL_FRONTEND_ORIGIN`: Allowed loopback origins (`http://localhost:5173`)

---

## 4. Database Setup

1. **Local MongoDB:**
   ```bash
   mongod --dbpath /path/to/data/db
   ```
2. **MongoDB Atlas:**
   - Create a MongoDB Atlas cluster (M0 or dedicated).
   - In Network Access, allow your server IP or `0.0.0.0/0` (with strong user credentials).
   - Copy connection string to `MONGODB_URI` in `backend/.env`.

---

## 5. Development Startup Commands

Start services in separate terminal windows:

### Terminal 1: Central Backend API
```bash
npm run dev --workspace backend
# Runs on http://localhost:4000
```

### Terminal 2: Hospital-Local ML Service
```bash
# Activate virtual environment
.\.venv\Scripts\activate
python ml/service.py
# Runs on http://127.0.0.1:8000
```

### Terminal 3: Hospital & Admin Web UI
```bash
npm run dev --workspace frontend
# Runs on http://localhost:5173
```

---

## 6. Production Build Commands

Compile both frontend and backend bundles:

```bash
# Build backend TypeScript
npm run build --workspace backend

# Build frontend production bundle
npm run build --workspace frontend
```

---

## 7. Backend Startup Commands (Production)

```bash
# Start compiled backend in production mode
npm start --workspace backend
# Or directly:
# node backend/dist/server.js
```

---

## 8. Deployment Procedure

### Option A: Docker Compose (Demonstration & Prototyping)
```bash
docker compose up --build
```
This runs MongoDB, Backend API, Frontend static server, and 3 independent hospital ML nodes on isolated volumes.

### Option B: Cloud/Server Deployment (Linux / Kubernetes / VM)
1. **Frontend:** Deploy `frontend/dist` to an HTTPS web server (Nginx, Cloudflare Pages, Vercel, or AWS S3 + CloudFront).
2. **Central API:** Deploy `backend` container or Node.js service behind a reverse proxy (Nginx or AWS ALB) with SSL termination.
3. **Artifact Storage:** Ensure `MODEL_ARTIFACT_ROOT` is mounted to a persistent volume (EBS, persistent volume claim, or Azure Disk).
4. **Hospital Nodes:** Each hospital deploys `ml/service.py` locally inside their hospital firewall or intranet. No inbound public ports required for hospitals.

---

## 9. Deployment Smoke-Test Checklist

- [ ] **Liveness & Readiness:**
  - `GET http://localhost:4000/api/health` returns `200 OK` with operational status.
  - `GET http://localhost:4000/api/ready` returns `200 OK` confirming database and artifact storage.
- [ ] **Hospital ML Health:**
  - `GET http://127.0.0.1:8000/health` returns `200 OK` with enabled tasks.
- [ ] **Authentication:**
  - Administrator logs in successfully at `/login`.
  - Hospital account registers and signs in.
- [ ] **Federation Round:**
  - Admin creates a round for Pneumonia/Retinopathy selecting registered hospital.
  - Round reflects in Hospital dashboard under "Federated round".
- [ ] **Local Training & Transmission:**
  - Hospital prepares dataset and runs local training.
  - Parameters package into AVM1 format and submit to central coordinator.
  - Admin dashboard shows update under "Confirmed model updates" with hospital name, round, and sample count.
- [ ] **Federated Aggregation:**
  - Coordinator aggregates updates using FedAvg and creates new global model version.
  - Participating hospitals can download the updated global model.
- [ ] **Clinical Inference:**
  - Local image prediction operates offline on `127.0.0.1:8000/predict` and displays classification without uploading images to central server.
