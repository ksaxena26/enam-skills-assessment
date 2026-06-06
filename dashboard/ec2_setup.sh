#!/bin/bash
# EC2 setup script — run once after first SSH login on a fresh Amazon Linux 2023 / Ubuntu instance.
# Tested on t3.small (2 vCPU, 2 GB RAM).

set -e

# ── 1. System packages ─────────────────────────────────────────────────────────
sudo apt-get update -y                          # Ubuntu
# sudo yum update -y                            # Amazon Linux — uncomment instead

sudo apt-get install -y git python3-pip python3-venv tmux

# ── 2. Clone / copy project ────────────────────────────────────────────────────
# Option A: clone from a private GitHub repo (set up SSH key first)
#   git clone git@github.com:<your-org>/enam-skills-assessment.git /opt/trading
#
# Option B: SCP the full directory from your local machine:
#   scp -r -i your-key.pem ./enam-skills-assessment ec2-user@<EC2_IP>:/opt/trading
#
# This script assumes the project is already at /opt/trading
PROJECT=/opt/trading

# ── 3. Python virtual environment ──────────────────────────────────────────────
python3 -m venv $PROJECT/.venv
source $PROJECT/.venv/bin/activate

pip install --upgrade pip
pip install -r $PROJECT/dashboard/requirements.txt
# Install any engine-specific packages not in requirements.txt:
pip install pandas_ta optuna optuna-dashboard pyarrow kaleido

# ── 4. Environment variable for dashboard password ────────────────────────────
# Add to ~/.bashrc so it persists across reboots:
echo 'export DASHBOARD_PASSWORD="changeme"' >> ~/.bashrc
source ~/.bashrc

# ── 5. Run dashboard persistently with tmux ───────────────────────────────────
tmux new-session -d -s dashboard \
  "source $PROJECT/.venv/bin/activate && \
   cd $PROJECT && \
   DASHBOARD_PASSWORD=changeme \
   streamlit run dashboard/app.py \
     --server.port 8501 \
     --server.address 0.0.0.0 \
     --server.headless true"

echo ""
echo "Dashboard started in tmux session 'dashboard'."
echo "Access at:  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
echo ""
echo "To attach:  tmux attach -t dashboard"
echo "To restart: tmux kill-session -t dashboard && bash $PROJECT/dashboard/ec2_setup.sh"
