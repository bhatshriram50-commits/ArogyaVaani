# Security

ArogyaVaani is a research prototype, not a certified medical product. It does not claim clinical validation, differential privacy, secure aggregation, or a formal guarantee against model-update leakage.

The central API uses bcrypt hashes, provisioned admin credentials, expiring JWTs, server-side Google ID-token verification, role/ownership checks, rate limits, Helmet headers, request limits, configured CORS, and generic error responses. Production requires MongoDB, persistent artifact storage, a 32+ character JWT secret, and HTTPS-only frontend origins.

The local ML service validates rooted paths, upload names/extensions/content, image dimensions, dataset limits, checkpoint task metadata, AVM1 tensors, and checksums. Keep it on loopback or private networking. Remote access needs HTTPS, firewall/allow-list rules, and `LOCAL_SERVICE_TOKEN`; a browser-visible token is not a replacement for network isolation.

Never centralize image upload, datasets, patient records, or prediction inputs. Before deployment replace the responsible-disclosure placeholder `security@example.invalid` with an owned contact; do not disclose vulnerabilities in public issue trackers.
