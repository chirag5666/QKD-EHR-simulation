import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION PARAMETERS ---
# These can be adjusted to test different scenarios
NUM_BITS = 4096  # Number of qubits to send in one transaction
EHR_DATA_SIZE_KB = 256 # Size of the Electronic Health Record to simulate transferring
QBER_THRESHOLD = 0.05 # Abort protocol if QBER exceeds this (e.g., 5%)
LATENCY_PER_KM_MS = 0.005 # Latency for light in fiber optics (approx. 5 microseconds/km)
PROCESSING_OVERHEAD_MS = 50 # Base latency for processing, sifting, etc.

class QuantumChannel:
    """
    Simulates the quantum channel.
    As per the project scope, this is an "idealized quantum channel with
    parameters for adjustable noise"[cite: 87]. It is not a physical simulation.
    """
    def __init__(self, distance, noise_level=0.01):
        self.distance = distance
        # Simple model: noise increases slightly with distance (attenuation)
        self.base_noise = noise_level
        self.noise = self.base_noise + (self.distance / 1000) # Add a small distance factor
        if self.noise > 1.0: self.noise = 1.0

    def transmit(self, qubits, eavesdropper_present=False):
        """Transmits qubits, applying noise and simulating an eavesdropper."""
        transmitted_qubits = []
        for bit, basis in qubits:
            # 1. Eavesdropper ("Eve") intercepts and measures
            if eavesdropper_present:
                eve_basis = np.random.randint(0, 2)
                if basis != eve_basis:
                    # Eve's measurement in the wrong basis randomizes the bit
                    bit = np.random.randint(0, 2)

            # 2. Apply channel noise
            if np.random.random() < self.noise:
                # Flip the bit to simulate a quantum bit error
                bit = 1 - bit

            transmitted_qubits.append((bit, basis))
        return transmitted_qubits

class Node:
    """
    Represents a healthcare entity (hospital, data center) in the QKD network.
    This class contains the logic for the BB84 protocol.
    """
    def __init__(self, name):
        self.name = name
        self.bits = None
        self.bases = None
        self.sifted_key = None

    def _generate_bits_and_bases(self, num_bits):
        """Generates random bits and bases for the QKD protocol."""
        self.bits = np.random.randint(0, 2, num_bits)
        self.bases = np.random.randint(0, 2, num_bits) # 0 for Rectilinear (+), 1 for Diagonal (x)

    def send_qubits(self, num_bits):
        """Encodes bits into qubits based on chosen bases."""
        self._generate_bits_and_bases(num_bits)
        # In a real system, this would be sending polarized photons.
        # Here, we represent a qubit as a tuple of (bit, basis).
        return list(zip(self.bits, self.bases))

    def receive_qubits(self, qubits):
        """Measures incoming qubits with a new set of random bases."""
        num_received = len(qubits)
        self._generate_bits_and_bases(num_received)
        measured_bits = []
        for i in range(num_received):
            received_bit, sender_basis = qubits[i]
            if self.bases[i] == sender_basis:
                # Bases match, measurement is accurate
                measured_bits.append(received_bit)
            else:
                # Bases mismatch, measurement is random (50/50 chance for 0 or 1)
                measured_bits.append(np.random.randint(0, 2))
        self.bits = np.array(measured_bits)


def run_bb84_simulation(distance, noise, num_bits, eavesdropper=False, verbose=True):
    """
    Orchestrates a single run of the BB84 QKD simulation.
    This function simulates the entire process from key generation to EHR transmission.
    """
    if verbose:
        print("-" * 50)
        print(f"Starting QKD Simulation: Distance={distance}km, Noise={noise:.2%}, Eavesdropper={eavesdropper}")
        print("-" * 50)

    start_time = time.time()

    # 1. Setup Network Entities
    alice = Node("Hospital A")
    bob = Node("Central Repository")
    channel = QuantumChannel(distance=distance, noise_level=noise)

    # 2. Quantum Phase: Qubit Transmission
    qubits_to_send = alice.send_qubits(num_bits)
    if verbose: print(f"[ALICE]   Generated and sent {len(qubits_to_send)} qubits.")
    
    transmitted_qubits = channel.transmit(qubits_to_send, eavesdropper_present=eavesdropper)
    if verbose: print(f"[CHANNEL] Qubits transmitted over {distance}km.")

    bob.receive_qubits(transmitted_qubits)
    if verbose: print(f"[BOB]     Measured {len(transmitted_qubits)} qubits.")

    # 3. Public Discussion Phase (Simulated)
    # Alice and Bob compare bases. In a real network, this happens over an authenticated classical channel.
    matching_bases_indices = np.where(alice.bases == bob.bases)[0]
    
    alice_sifted_key = alice.bits[matching_bases_indices]
    bob_sifted_key = bob.bits[matching_bases_indices]
    sifted_key_length = len(alice_sifted_key)

    if verbose: print(f"[SIFTING] Bases compared. Sifted key length: {sifted_key_length} bits.")
    
    if sifted_key_length == 0:
        return {"qber": 1, "skr": 0, "latency": 0, "success": False}

    # 4. Parameter Estimation: Calculate QBER
    # A random subset of the sifted key is used to estimate the error rate.
    sample_size = min(sifted_key_length // 2, 500) # Use half the key or max 500 bits for estimation
    sample_indices = np.random.choice(sifted_key_length, sample_size, replace=False)
    
    alice_sample = alice_sifted_key[sample_indices]
    bob_sample = bob_sifted_key[sample_indices]
    
    errors = np.sum(alice_sample != bob_sample)
    qber = errors / sample_size if sample_size > 0 else 0
    
    if verbose: print(f"[QBER]    Estimated QBER from {sample_size}-bit sample: {qber:.2%}")

    # Remove the sample bits from the key
    final_key_indices = np.setdiff1d(np.arange(sifted_key_length), sample_indices)
    alice_final_key = alice_sifted_key[final_key_indices]

    # 5. Key Reconciliation and Privacy Amplification (Simulated)
    # If QBER is too high, abort. This detects the eavesdropper.
    if qber >= QBER_THRESHOLD:
        if verbose: print(f"[ABORT]   QBER ({qber:.2%}) exceeds threshold ({QBER_THRESHOLD:.2%}). Key discarded!")
        return {"qber": qber, "skr": 0, "latency": 0, "success": False}

    # The length of the final key is reduced based on the QBER.
    # This simulates error correction and privacy amplification.
    # A common approximation is to reduce the key length by a factor related to the Shannon entropy.
    # For simplicity, we use a linear factor reduction. `leaked_info = sifted_key_length * qber`.
    final_key_length = len(alice_final_key) * (1 - 2 * qber) # A simple but effective approximation
    final_key_length = max(0, int(final_key_length))
    
    qkd_end_time = time.time()
    qkd_latency_ms = (qkd_end_time - start_time) * 1000 + PROCESSING_OVERHEAD_MS
    
    # Calculate Secure Key Rate (SKR) in kbps
    skr_kbps = (final_key_length / (qkd_latency_ms / 1000)) / 1000 if qkd_latency_ms > 0 else 0

    if verbose: print(f"[SUCCESS] Final secure key length: {final_key_length} bits.")
    if verbose: print(f"[SKR]     Secure Key Rate: {skr_kbps:.2f} kbps.")

    # 6. Simulated EHR Transmission
    # Now use the established key to encrypt and send data
    classical_latency_ms = distance * LATENCY_PER_KM_MS
    total_transaction_latency = qkd_latency_ms + classical_latency_ms
    
    if verbose:
        print("\n--- EHR Transaction Simulation ---")
        print(f"QKD Establishment Latency: {qkd_latency_ms:.2f} ms")
        print(f"Classical Data Tx Latency: {classical_latency_ms:.2f} ms")
        print(f"Total Latency for {EHR_DATA_SIZE_KB}KB EHR: {total_transaction_latency:.2f} ms")

    return {
        "qber": qber,
        "skr": skr_kbps,
        "latency": total_transaction_latency,
        "success": True
    }


def generate_plots():
    """
    
    """
    print("\n" + "="*60)
    print("Generating data for performance analysis plots...")
    print("="*60)

    # --- Plot 1: QBER and SKR vs. Channel Noise ---
    noise_levels = np.linspace(0, 0.1, 20) # 0% to 10% noise
    qber_results = []
    skr_results = []

    for noise in noise_levels:
        res = run_bb84_simulation(distance=20, noise=noise, num_bits=NUM_BITS, eavesdropper=False, verbose=False)
        qber_results.append(res["qber"] * 100) # in percentage
        skr_results.append(res["skr"])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Channel Noise Level (%)')
    ax1.set_ylabel('Quantum Bit Error Rate (QBER) (%)', color=color)
    ax1.plot(noise_levels * 100, qber_results, 'o-', color=color, label='QBER')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=QBER_THRESHOLD * 100, color='gray', linestyle='--', label=f'Abort Threshold ({QBER_THRESHOLD*100}%)')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Secure Key Rate (SKR) (kbps)', color=color)
    ax2.plot(noise_levels * 100, skr_results, 's--', color=color, label='SKR')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Performance vs. Channel Noise (Distance: 20km)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    plt.savefig('performance_vs_noise.png')
    print("Saved plot: performance_vs_noise.png")

    # --- Plot 2: SKR vs. Distance ---
    distances = np.linspace(5, 80, 15) # 5km to 80km
    skr_vs_dist_results = []
    for dist in distances:
        res = run_bb84_simulation(distance=dist, noise=0.01, num_bits=NUM_BITS, eavesdropper=False, verbose=False)
        skr_vs_dist_results.append(res["skr"])

    plt.figure(figsize=(10, 6))
    plt.plot(distances, skr_vs_dist_results, 'o-')
    plt.xlabel('Distance (km)')
    plt.ylabel('Secure Key Rate (SKR) (kbps)')
    plt.title('Secure Key Rate vs. Distance (Noise: 1%)')
    plt.grid(True)
    plt.savefig('skr_vs_distance.png')
    print("Saved plot: skr_vs_distance.png")

    plt.show()


if __name__ == '__main__':
    # Run a single detailed simulation to show the step-by-step process
    run_bb84_simulation(distance=20, noise=0.02, num_bits=NUM_BITS, eavesdropper=False, verbose=True)

    # Run a simulation demonstrating eavesdropper detection
    run_bb84_simulation(distance=20, noise=0.02, num_bits=NUM_BITS, eavesdropper=True, verbose=True)

    # Generate and save the performance plots for your report
    generate_plots()