# Conceptual Framework and Simulation of QKD for Secure Electronic Health Records (EHRs)

**Authors:** Chirag MV Gowda , Yash Tandon

## Abstract

The use of Electronic Health Records (EHRs) has transformed the delivery of healthcare through unprecedented data-sharing efficiency and patient care coordination. However, this digital revolution has brought with it the challenges of substantial security and privacy risks. Centralized EHR systems are under threat from single-point-of-failure attacks, while the confidentiality of medical information renders them a prime target for cyberattackers. The forthcoming arrival of quantum computation presents a calamitous risk to the cryptographic underpinnings of existing EHR security, as quantum algorithms can effectively compromise the asymmetric encryption standards that secure this sensitive infrastructure.

This project proposes and simulates a novel Conceptual Framework for the protection of EHRs through Quantum Key Distribution (QKD), an information-theoretically secure technology founded on the intrinsic principles of quantum mechanics.We present a secure, decentralized model of a QKD-supported EHR network designed as a trusted-node metropolitan-area network. This repository contains a comprehensive suite of Python-based simulations to test the feasibility and performance of this framework by analyzing key parameters such as Secure Key Rate (SKR), Quantum Bit Error Rate (QBER), and network latency under various operational scenarios. The results affirm that a QKD-based solution offers a feasible and future-proof measure for protecting the confidentiality and integrity of EHRs against both classical and quantum attack.

## Key Features

-   **Comprehensive Analysis Suite:** A Python script (`main_simulation_suite.py`) that runs detailed simulations and generates publication-quality graphs for:
    -   3D Performance Envelope (SKR vs. Distance & Noise)
    -   Dynamic Key Pool Throughput Analysis
    -   Protocol Robustness (BB84 vs. B92) under attack
-   **Interactive Network Dashboard:** A Streamlit web application (`qkd_dashboard_app.py`) for live, visual demonstrations of the network's operation, security status, and real-time attack detection.
-   **Qiskit Proof-of-Concept:** A script (`qiskit_bb84_demo.py`) to demonstrate the low-level quantum mechanics of the BB84 protocol and eavesdropper detection.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/qkd-ehr-simulation.git](https://github.com/your-username/qkd-ehr-simulation.git)
    cd qkd-ehr-simulation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Simulations

### 1. Main Analysis Suite

This script runs all core analyses and saves the resulting graphs to the `results/` folder.

```bash
python src/main_simulation_suite.py
