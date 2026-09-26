# ArogyaVaani

Privacy-preserved medical AI for collaborative hospital learning. Hospital images remain inside the local Python training process; the central API receives model parameters and training metadata only.

## Structure
- `frontend`: React + Vite responsive application
- `backend`: Express + TypeScript REST API, MongoDB-ready persistence boundary
- `ml`: PyTorch DenseNet local training and FedAvg service

## Run
1. Install Node.js 20+ and Python 3.11+.
2. Run `npm install` at the repository root.
3. Copy `backend/.env.example` to `backend/.env` and configure MongoDB, JWT, and Google OAuth credentials. For Google Identity Services, also copy `frontend/.env.example` to `frontend/.env` and use the same web client ID.
4. Install ML dependencies with `python -m pip install -r ml/requirements.txt`.
5. Run `npm run dev` for the frontend and backend. Run `python ml/service.py` for the local training service.

The frontend defaults to `http://localhost:4000` for the coordination API and `http://127.0.0.1:8000` for the hospital-local ML service. Override these with `VITE_API_URL` and `VITE_LOCAL_ML_URL` in `frontend/.env` when deploying. Model updates can be large; adjust `MODEL_UPDATE_MAX_BYTES` in `backend/.env` only when required by the chosen architecture.

Google OAuth uses Google Identity Services in the browser and verifies every ID token on the backend against `GOOGLE_CLIENT_ID`. New Google users complete hospital onboarding with their hospital name. Administrator accounts are provisioned only from `ADMIN_EMAIL` and `ADMIN_PASSWORD` on the server; there is no admin signup route.
