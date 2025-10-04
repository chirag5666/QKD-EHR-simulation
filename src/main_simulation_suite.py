# main_simulation_suite.py

import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# --- GLOBAL SIMULATION CONFIGURATION ---
NUM_BITS_FOR_KEY_ESTABLISHMENT = 8192
QBER_THRESHOLD = 0.05
BASE_NOISE_LEVEL = 0.01

# --- CORE QKD COMPONENTS ---

class QuantumChannel:
    """Simulates the quantum channel with noise and a potential eavesdropper."""
    def __init__(self, distance):
        self.distance = distance
        self.noise = BASE_NOISE_LEVEL + (distance / 1000)

    def transmit(self, qubits, eavesdropper_present=False, interception_rate=1.0):
        transmitted_qubits = []
        for bit, basis in qubits:
            if eavesdropper_present and np.random.random() < interception_rate:
                eve_basis = np.random.randint(0, 2)
                if basis != eve_basis: bit = np.random.randint(0, 2)
            if np.random.random() < self.noise: bit = 1 - bit
            transmitted_qubits.append((bit, basis))
        return transmitted_qubits

class Node:
    """Represents a node in the QKD network."""
    def __init__(self, name):
        self.name = name
        self.bits = None
        self.bases = None

    def prepare_qubits(self, num_bits):
        self.bits = np.random.randint(0, 2, num_bits)
        self.bases = np.random.randint(0, 2, num_bits)
        return list(zip(self.bits, self.bases))

    def measure_qubits(self, qubits):
        self.bases = np.random.randint(0, 2, len(qubits))
        measured_bits = []
        for i, (received_bit, sender_basis) in enumerate(qubits):
            if self.bases[i] == sender_basis: measured_bits.append(received_bit)
            else: measured_bits.append(np.random.randint(0, 2))
        self.bits = np.array(measured_bits)

# --- LINK-LEVEL SIMULATION FUNCTIONS ---

def run_qkd_link_simulation(sender, receiver, channel, num_bits, protocol="BB84", eavesdropper_present=False, interception_rate=1.0):
    """Orchestrates a single QKD key exchange for BB84 or B92 and returns the results."""
    # Protocol-specific preparation
    if protocol == "BB84":
        qubits_sent = sender.prepare_qubits(num_bits)
    elif protocol == "B92":
        sender.bits = np.random.randint(0, 2, num_bits)
        sender.bases = sender.bits
        qubits_sent = list(zip(sender.bits, sender.bases))
    else:
        return {"success": False, "qber": 1, "skr": 0}

    qubits_received = channel.transmit(qubits_sent, eavesdropper_present, interception_rate)
    
    # Sifting
    sifted_key_sender = []
    sifted_key_receiver = []
    if protocol == "BB84":
        receiver.measure_qubits(qubits_received)
        matching_indices = np.where(sender.bases == receiver.bases)[0]
        sifted_key_sender = sender.bits[matching_indices]
        sifted_key_receiver = receiver.bits[matching_indices]
    elif protocol == "B92":
        for i, (received_bit, sender_basis) in enumerate(qubits_received):
            bob_measurement_basis = np.random.randint(0, 2)
            if bob_measurement_basis != sender_basis:
                measured_bit = np.random.randint(0, 2)
                if measured_bit != sender_basis:
                    sifted_key_sender.append(sender.bits[i])
                    sifted_key_receiver.append(received_bit)

    sifted_key_sender = np.array(sifted_key_sender)
    sifted_key_receiver = np.array(sifted_key_receiver)

    if len(sifted_key_sender) < 40: return {"success": False, "qber": 1, "skr": 0}

    # QBER Estimation & Final Key
    sample_size = min(len(sifted_key_sender) // 2, 500)
    sample_indices = np.random.choice(len(sifted_key_sender), sample_size, replace=False)
    errors = np.sum(sifted_key_sender[sample_indices] != sifted_key_receiver[sample_indices])
    qber = errors / sample_size if sample_size > 0 else 0
    if qber >= QBER_THRESHOLD: return {"success": False, "qber": qber, "skr": 0}
    
    remaining_key_length = len(sifted_key_sender) - sample_size
    final_key_length = int(remaining_key_length * (1 - 2 * qber))
    simulated_time_s = 0.2 # Assume a fixed time for exchange for SKR calculation
    skr_bps = final_key_length / simulated_time_s if simulated_time_s > 0 else 0
    
    return {"success": True, "qber": qber, "skr": skr_bps}

# --- ADVANCED SIMULATION & GRAPHING FUNCTIONS ---

def run_3d_surface_analysis():
    print("Running 3D Performance Surface Analysis...")
    distances = np.linspace(5, 80, 20)
    added_noise_levels = np.linspace(0, 0.04, 20)
    X, Y = np.meshgrid(distances, added_noise_levels)
    Z = np.zeros(X.shape)
    sender, receiver = Node("A"), Node("B")
    for i in range(len(distances)):
        for j in range(len(added_noise_levels)):
            dist, noise = X[j, i], Y[j, i]
            channel = QuantumChannel(distance=dist)
            channel.noise += noise
            result = run_qkd_link_simulation(sender, receiver, channel, 4096)
            if result["success"]: Z[j, i] = result['skr']
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y*100, Z, cmap='viridis', edgecolor='none')
    ax.set_title('QKD Performance Envelope (SKR vs. Distance & Noise)')
    ax.set_xlabel('Distance (km)'); ax.set_ylabel('Added Environmental Noise (%)'); ax.set_zlabel('Secure Key Rate (bits/sec)')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='SKR (bps)')
    plt.savefig('3d_performance_surface.png')
    print("✅ Graph '3d_performance_surface.png' saved.")

def run_dynamic_key_pool_analysis():
    print("Running Dynamic Key Pool Analysis...")
    link_skr = {"Vellore_Hospital_Link": 8000, "CMC_Lab_Link": 5000, "Katpadi_Clinic_Link": 12000}
    key_pool = {name: 0 for name in link_skr.keys()}
    bits_per_request = 2048
    history = []
    for t_ms in range(0, 10000, 10):
        for name, skr in link_skr.items(): key_pool[name] += skr * 0.01
        if t_ms % 100 == 0:
            requester = np.random.choice(list(link_skr.keys()))
            if key_pool[requester] >= bits_per_request: key_pool[requester] -= bits_per_request
        history.append(key_pool.copy())
    df_history = pd.DataFrame(history)
    # Plotting
    plt.figure(figsize=(12, 7))
    for column in df_history.columns: plt.plot(df_history.index * 10, df_history[column], label=f'{column}')
    plt.title('Dynamic Key Pool Size Under Medium Traffic Load'); plt.xlabel('Time (milliseconds)'); plt.ylabel('Available Secure Key Bits in Pool')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('key_pool_simulation.png')
    print("✅ Graph 'key_pool_simulation.png' saved.")

def run_robustness_analysis():
    print("Running Protocol Robustness Analysis...")
    interception_rates = np.linspace(0, 1, 15)
    qber_results = {"BB84": [], "B92": []}
    sender, receiver, channel = Node("A"), Node("B"), QuantumChannel(distance=20)
    for protocol in ["BB84", "B92"]:
        for rate in interception_rates:
            result = run_qkd_link_simulation(sender, receiver, channel, 8192, protocol=protocol, eavesdropper_present=True, interception_rate=rate)
            qber_results[protocol].append(result['qber'])
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(interception_rates * 100, qber_results['BB84'], 'o-', label='BB84 Protocol', color='royalblue')
    plt.plot(interception_rates * 100, qber_results['B92'], 's--', label='B92 Protocol', color='lightcoral')
    plt.axhline(y=QBER_THRESHOLD, color='red', linestyle=':', label=f'Detection Threshold ({QBER_THRESHOLD:.0%})')
    plt.title('Protocol Robustness to Eavesdropping'); plt.xlabel('Attacker Interception Rate (%)'); plt.ylabel('Measured QBER')
    plt.legend(); plt.grid(True); plt.ylim(0, 0.5)
    plt.savefig('robustness_comparison.png')
    print("✅ Graph 'robustness_comparison.png' saved.")

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    run_3d_surface_analysis()
    run_dynamic_key_pool_analysis()
    run_robustness_analysis()
    
    print("\nAll simulations complete. Showing plots...")
    plt.show()