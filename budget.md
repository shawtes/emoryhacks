# Hackathon Demo Budget (Carter Plan)

## Key Assumptions
- Demo window: 1 month on Firebase Blaze plan (required for Cloud Functions + Storage).
- Personas: 5 clinics/clinic admins (seeded), 5 doctors, 20 active patients.
- Audio workload: each patient uploads 5 recordings at ~5 MB each (≈0.5 GB total stored), doctors replay each recording twice.
- App traffic limited to staging/demo sessions; no push notifications or third-party APIs beyond Firebase.
- All resources deployed in the same region (e.g., `us-central1`) to avoid inter-region egress fees.

## Estimated Firebase Costs

| Service | Usage Snapshot | Unit Rate (typical) | Estimated Cost |
| --- | --- | --- | --- |
| **Cloud Firestore** | ~15k reads, 5k writes, 1k deletes, <1 GB stored | Reads $0.06/100k, Writes $0.18/100k, Deletes $0.02/100k, Storage $0.026/GB-mo | **≈ $0.05** (all operations within free tier; storage under 1 GB) |
| **Firebase Storage (GCS)** | 0.5 GB stored, 5 GB downloads (doctors replay), ~2k ops | Storage $0.026/GB-mo, Egress $0.12/GB, Ops $0.05/10k (Class A), $0.004/10k (Class B) | **≈ $0.50** storage + **$0.60** egress + negligible ops ≈ **$1.10** |
| **Cloud Functions** | 10k invocations (register, stubs, activation, audio tokens) with avg 256 MB RAM & 500 ms runtime | Free quota: 2M invocations, 400k GB‑s, 200k CPU‑s | **$0.00** (well within free tier) |
| **Firebase Authentication** | 30 MAU email/password | Free up to 50k MAU | **$0.00** |
| **Firebase Hosting / Emulator traffic** | Optional for demo landing page; assume <1 GB egress | Free tier: 10 GB storage, 360 MB/day egress | **$0.00** (stay under free allotment) |
| **Cloud Logging/Monitoring** | Default ingest from Functions | Free up to 50 GiB/month | **$0.00** |

## Total Estimated Spend
- **≈ $1.20 for the entire hackathon demo month** (dominated by Storage bytes + download egress). Everything else remains in Firebase’s always-free quotas at this scale.

## Safety Margin & Notes
- Add a 5× buffer (≈ $6) if you expect more audio or repeated demos; still negligible.
- If you introduce phone auth, multi-region storage, or keep audio long-term, costs will grow accordingly.
- Staying within emulators for most dev/testing keeps usage near zero until you present the demo.
- Remember Blaze plan requires attaching a billing method even if the calculated spend rounds to $0.


