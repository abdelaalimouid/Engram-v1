#!/usr/bin/env bash
#
# One-shot deploy of Engram to a fresh Alibaba Cloud ECS instance (Ubuntu 22.04).
# Paste this whole thing into the ECS Workbench terminal, or run:
#   bash <(curl -fsSL https://raw.githubusercontent.com/abdelaalimouid/Engram-v1/main/scripts/deploy_ecs.sh)
#
# It will ask for your Qwen Cloud (DashScope) API key, then start the server on :8000.

set -euo pipefail

REPO="https://github.com/abdelaalimouid/Engram-v1.git"
DIR="$HOME/Engram-v1"

echo "==> Installing system packages (python3, venv, git)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git

echo "==> Cloning repo"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone "$REPO" "$DIR"
fi
cd "$DIR"

echo "==> Python virtualenv + dependencies"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip -q
./.venv/bin/pip install -r requirements.txt -q

if [ ! -f .env ]; then
  echo
  read -r -p "Paste your Qwen Cloud / DashScope API key (sk-...): " KEY
  echo "DASHSCOPE_API_KEY=${KEY}" > .env
  echo "==> Wrote .env"
fi

# Public IP for the "open it in a browser" step.
PUBIP="$(curl -fsSL https://100.100.100.200/latest/meta-data/eipv4 2>/dev/null || curl -fsSL ifconfig.me 2>/dev/null || echo '<your-ecs-public-ip>')"

echo
echo "======================================================================"
echo " Starting Engram on Alibaba Cloud ECS"
echo " Local:  http://0.0.0.0:8000"
echo " Public: http://${PUBIP}:8000   (open this in your browser)"
echo " Make sure the ECS security group allows inbound TCP 8000."
echo "======================================================================"
echo
exec ./.venv/bin/uvicorn engram.server:app --host 0.0.0.0 --port 8000
