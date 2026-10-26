# DVD Project: ML-driven PVT-aware leakage estimation

Digital VLSI Design course project, IIIT Hyderabad. Team 22: Aryan Agrawal, Manikya Pant, Vansh Jaiswal, Saarthak Sabharwal.

Model `45nm_MGK.pm`, nominal VDD 0.9 V, L = 45 nm, test circuit: 8-bit parity checker. Simulator: ngspice 42.

This first part sets up the single-transistor simulator and generates the Part A datasets (4 cases x 10,000 samples) as the official project document (`DVD_ML_Project.pdf`) specifies.

## Run

    sudo apt install -y ngspice
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.lock.txt
    python scripts/verify_sim.py          # simulator checks
    python scripts/a1_generate_data.py    # Part A datasets -> data/partA/
    python scripts/a2_check_data.py       # data check -> reports/partA/DATA_CHECK.md
