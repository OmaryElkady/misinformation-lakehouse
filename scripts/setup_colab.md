# Colab Training Setup Guide

Step-by-step instructions for fine-tuning RoBERTa on Google Colab
using the local Gold Delta table and MLflow tracking server.

---

## Prerequisites

- Local MLflow server running (via `docker compose up -d`)
- ngrok account — free tier is enough (sign up at https://ngrok.com)
- Google account with Google Drive

---

## Step 1 — Export Gold Data to Parquet

Run this locally to read the Gold Delta table and write the train/val splits to disk:

```python
# From the project root (with venv active):
python - <<'EOF'
from src.training.train import export_gold_to_parquet
train_path, val_path = export_gold_to_parquet()
print(f"Train: {train_path}")
print(f"Val:   {val_path}")
EOF
```

This writes two files:
- `./data/exports/train.parquet`
- `./data/exports/val.parquet`

---

## Step 2 — Upload Parquets to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Create a folder called `misinfo` inside `My Drive`
3. Upload both parquet files into `My Drive/misinfo/`

Your Drive layout should look like:
```
My Drive/
  misinfo/
    train.parquet
    val.parquet
```

---

## Step 3 — Expose Local MLflow via ngrok

Colab runs in Google's cloud and cannot reach `localhost` on your machine directly.
ngrok creates a public HTTPS tunnel that forwards requests to your local MLflow server.

**Install ngrok** (if not already installed):

```bash
# macOS
brew install ngrok

# Linux / WSL2
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok
```

**Authenticate** (one-time setup — paste your token from the ngrok dashboard):

```bash
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
```

You can store the token in `.env`:
```
NGROK_AUTHTOKEN=your_ngrok_token_here
```

**Start the tunnel** (keep this terminal open while Colab is running):

```bash
ngrok http 5000 --host-header="localhost:5000"
```

The `--host-header` flag is required — MLflow 2.15+ rejects requests whose Host header
doesn't match `localhost`. ngrok by default forwards the public hostname
(`abc123.ngrok-free.dev`) as the Host header, which MLflow blocks as a DNS-rebinding
attack. This flag rewrites it to `localhost:5000` before forwarding.

You will see output like:
```
Forwarding   https://abc123.ngrok-free.app -> http://localhost:5000
```

Copy the `https://...ngrok-free.app` URL — you will paste it into Cell 2 of the notebook.

---

## Step 4 — Open the Notebook in Colab and Run All Cells

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `notebooks/colab_training.ipynb`
3. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU → Save**
4. In **Cell 2**, replace `YOUR_NGROK_URI_HERE` with your ngrok HTTPS URL
5. **Run Cell 1 only first**, then do **Runtime → Restart session** (Ctrl+M .)
   - This is required — the old numpy binary stays loaded in memory until restart.
     Skipping the restart causes `ValueError: numpy.dtype size changed` even after a correct install.
   - After restarting, do NOT re-run Cell 1. Run Cell 2 onward.
6. Run the remaining cells in order

Training takes roughly 15–30 minutes on a T4 GPU for the full dataset.

**Viewing MLflow during training:** open `http://localhost:5000` in your local browser
(not the ngrok URL — MLflow 2.15+ CSRF protection blocks the UI when accessed via ngrok).
Look for the **misinformation-detection** experiment in the left panel.

---

## Step 5 — Download the Trained Model from Drive

After Cell 8 completes, the model is saved at `My Drive/misinfo/model/`.

Download it locally:

1. In Google Drive, right-click `misinfo/model/` → **Download**
2. Unzip and move the contents to `./data/models/roberta-v1/`:

```bash
mkdir -p ./data/models/roberta-v1
# After unzipping the downloaded folder:
mv ~/Downloads/model/* ./data/models/roberta-v1/
```

---

## Step 6 — Register the Model Locally

Copy the `run_id` printed at the end of Cell 7, then run locally:

```python
from src.training.train import register_model

register_model("PASTE_YOUR_RUN_ID_HERE")
```

This will:
- Register the model in the MLflow Model Registry under `misinformation-roberta-v1`
- Promote it to **Production** if `eval_f1 >= 0.80`
- Otherwise send it to **Staging** for review

Check the result at `http://localhost:5000` → **Models** tab.
