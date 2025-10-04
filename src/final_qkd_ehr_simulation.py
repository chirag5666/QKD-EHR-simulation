import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import math

# --- --------------------------------- ---
# --- GLOBAL SIMULATION CONFIGURATION ---
# --- --------------------------------- ---
NUM_BITS_FOR_KEY_ESTABLISHMENT = 8192  # Increased for more stable results
QBER_THRESHOLD = 0.05
LATENCY_PER_KM_MS = 0.005
HUB_PROCESSING_DELAY_MS = 20
BASE_NOISE_LEVEL = 0.01

# --- --------------------------- ---
# --- CORE QKD PROTOCOL CLASSES ---
# --- --------------------------- ---

class QuantumChannel:
    """Simulates the quantum channel, including noise and a potential eavesdropper."""
    def __init__(self, distance):
        self.distance = distance
        self.noise = BASE_NOISE_LEVEL + (distance / 1000)

    def transmit(self, qubits, eavesdropper_present=False):
        transmitted_qubits = []
        for bit, basis in qubits:
            if eavesdropper_present:
                eve_basis = np.random.randint(0, 2)
                if basis != eve_basis:
                    bit = np.random.randint(0, 2)
            if np.random.random() < self.noise:
                bit = 1 - bit
            transmitted_qubits.append((bit, basis))
        return transmitted_qubits

class Node:
    """Represents a node in the QKD network (e.g., hospital, repository)."""
    def __init__(self, name, location=(0,0)):
        self.name = name
        self.location = location # (x, y) coordinates for distance calculation
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
            if self.bases[i] == sender_basis:
                measured_bits.append(received_bit)
            else:
                measured_bits.append(np.random.randint(0, 2))
        self.bits = np.array(measured_bits)

# --- ------------------------------------- ---
# --- LINK-LEVEL & ARCHITECTURE SIMULATORS ---
# --- ------------------------------------- ---

def run_bb84_link_simulation(sender, receiver, channel, num_bits, eavesdropper_present=False):
    qubits_sent = sender.prepare_qubits(num_bits)
    qubits_received = channel.transmit(qubits_sent, eavesdropper_present=eavesdropper_present)
    receiver.measure_qubits(qubits_received)
    matching_bases_indices = np.where(sender.bases == receiver.bases)[0]
    sifted_key_sender = sender.bits[matching_bases_indices]
    sifted_key_receiver = receiver.bits[matching_bases_indices]
    if len(sifted_key_sender) < 40:
        return {"success": False, "qber": 1, "final_key_length": 0}
    sample_size = min(len(sifted_key_sender) // 2, 500)
    sample_indices = np.random.choice(len(sifted_key_sender), sample_size, replace=False)
    errors = np.sum(sifted_key_sender[sample_indices] != sifted_key_receiver[sample_indices])
    qber = errors / sample_size if sample_size > 0 else 0
    if qber >= QBER_THRESHOLD:
        return {"success": False, "qber": qber, "final_key_length": 0}
    remaining_key_length = len(sifted_key_sender) - sample_size
    final_key_length = int(remaining_key_length * (1 - 2 * qber))
    return {"success": True, "qber": qber, "final_key_length": max(0, final_key_length)}

def run_b92_link_simulation(sender, receiver, channel, num_bits, eavesdropper_present=False):
    """Simulates the B92 QKD protocol."""
    sender.bits = np.random.randint(0, 2, num_bits)
    sender.bases = sender.bits
    qubits_sent = list(zip(sender.bits, sender.bases))
    qubits_received = channel.transmit(qubits_sent, eavesdropper_present=eavesdropper_present)
    sifted_key_sender = []
    sifted_key_receiver = []
    for i, (received_bit, sender_basis) in enumerate(qubits_received):
        bob_measurement_basis = np.random.randint(0, 2)
        if bob_measurement_basis != sender_basis:
            measured_bit = np.random.randint(0, 2)
            if measured_bit != sender_basis:
                sifted_key_sender.append(sender.bits[i])
                sifted_key_receiver.append(received_bit)
    sifted_key_sender = np.array(sifted_key_sender)
    sifted_key_receiver = np.array(sifted_key_receiver)
    if len(sifted_key_sender) < 40:
        return {"success": False, "qber": 1, "final_key_length": 0}
    # (Rest of QBER estimation and key generation is the same as BB84)
    sample_size = min(len(sifted_key_sender) // 2, 500)
    sample_indices = np.random.choice(len(sifted_key_sender), sample_size, replace=False)
    errors = np.sum(sifted_key_sender[sample_indices] != sifted_key_receiver[sample_indices])
    qber = errors / sample_size if sample_size > 0 else 0
    if qber >= QBER_THRESHOLD: return {"success": False, "qber": qber, "final_key_length": 0}
    remaining_key_length = len(sifted_key_sender) - sample_size
    final_key_length = int(remaining_key_length * (1 - 2 * qber))
    return {"success": True, "qber": qber, "final_key_length": max(0, final_key_length)}

def simulate_pqc_ehr_transfer(source_node, dest_node, ehr_size_kb):
    """Simulates an EHR transfer using a PQC-based security model."""
    print(f"\n--- Simulating PQC-based EHR transfer for {ehr_size_kb}KB from '{source_node.name}' to '{dest_node.name}' ---")
    pqc_key_latency = 0.1 + 0.15 + 0.2  # KeyGen + Encap + Decap in ms (approximated from Kyber benchmarks)
    distance = math.dist(source_node.location, dest_node.location)
    classical_latency = distance * LATENCY_PER_KM_MS
    total_latency = pqc_key_latency + classical_latency
    print(f"1. PQC Key Exchange (computation). Time: {pqc_key_latency:.2f} ms")
    print(f"2. Data transfer {source_node.name} -> {dest_node.name} ({distance:.1f}km). Time: {classical_latency:.2f} ms")
    print(f"--- ✅ PQC COMPLETED: Total End-to-End Latency: {total_latency:.2f} ms ---")
    return total_latency

# --- ---------------- ---
# --- QKD_Network CLASS ---
# --- ---------------- ---

class QKD_Network:
    """Manages the trusted-node network, its topology, and all simulations."""
    def __init__(self, hub_name, hub_location=(0,0)):
        self.hub_name = hub_name
        self.nodes = {}
        self.channels = {}
        self.key_store = {}
        self.add_node(hub_name, hub_location)

    def add_node(self, node_name, location):
        self.nodes[node_name] = Node(node_name, location)

    def add_channel(self, node1_name, node2_name):
        distance = math.dist(self.nodes[node1_name].location, self.nodes[node2_name].location)
        channel_name = tuple(sorted((node1_name, node2_name)))
        self.channels[channel_name] = QuantumChannel(distance=distance)

    def establish_all_keys(self, protocol="BB84", attacked_link=None):
        hub_node = self.nodes[self.hub_name]
        print(f"\n--- Establishing Keys using {protocol} Protocol (Attacked Link: {attacked_link or 'None'}) ---")
        for spoke_name in self.nodes:
            if spoke_name == self.hub_name: continue
            is_attacked = tuple(sorted((self.hub_name, spoke_name))) == attacked_link
            spoke_node = self.nodes[spoke_name]
            channel = self.channels[tuple(sorted((self.hub_name, spoke_name)))]
            if protocol == "BB84": result = run_bb84_link_simulation(hub_node, spoke_node, channel, NUM_BITS_FOR_KEY_ESTABLISHMENT, is_attacked)
            elif protocol == "B92": result = run_b92_link_simulation(hub_node, spoke_node, channel, NUM_BITS_FOR_KEY_ESTABLISHMENT, is_attacked)
            else: continue
            if result["success"]:
                key_len = result["final_key_length"]
                self.key_store[(self.hub_name, spoke_name, protocol)] = key_len
                print(f"✅ {protocol} SUCCESS: Key between '{self.hub_name}' and '{spoke_name}'. Length: {key_len} bits")
            else:
                print(f"❌ {protocol} FAILED: Key establishment for '{spoke_name}'. (QBER: {result['qber']:.2%}) - ATTACK DETECTED!")

    def simulate_ehr_transfer(self, source_name, dest_name, ehr_size_kb):
        print(f"\n--- Attempting QKD-based EHR transfer of {ehr_size_kb}KB from '{source_name}' to '{dest_name}' ---")
        if (self.hub_name, source_name, "BB84") not in self.key_store or (self.hub_name, dest_name, "BB84") not in self.key_store:
            print("❌ ABORTED: Cannot perform transfer. A required secure key is missing.")
            return None
        channel1 = self.channels[tuple(sorted((source_name, self.hub_name)))]
        latency1 = channel1.distance * LATENCY_PER_KM_MS
        channel2 = self.channels[tuple(sorted((dest_name, self.hub_name)))]
        latency2 = channel2.distance * LATENCY_PER_KM_MS
        total_latency = latency1 + HUB_PROCESSING_DELAY_MS + latency2
        print(f"--- ✅ QKD COMPLETED: Total End-to-End Latency: {total_latency:.2f} ms ---")
        return total_latency

# --- --------------------- ---
# --- GRAPHING FUNCTIONS ---
# --- --------------------- ---

def plot_protocol_comparison(bb84_results, b92_results):
    labels = list(bb84_results.keys())
    bb84_keys = list(bb84_results.values())
    b92_keys = list(b92_results.values())
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, bb84_keys, width, label='BB84', color='royalblue')
    rects2 = ax.bar(x + width/2, b92_keys, width, label='B92', color='lightcoral')
    ax.set_ylabel('Total Secure Key Bits Generated')
    ax.set_title('Protocol Performance Comparison: BB84 vs. B92')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.savefig('protocol_comparison.png')
    print("\n✅ Graph 'protocol_comparison.png' has been saved.")

def plot_architecture_comparison(qkd_latency, pqc_latency, transfer_description):
    labels = ['QKD Trusted-Node', 'PQC Direct Link']
    latencies = [qkd_latency, pqc_latency]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, latencies, color=['orangered', 'dodgerblue'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f} ms', va='bottom', ha='center', fontsize=12)
    plt.ylabel('Total End-to-End Latency (ms)')
    plt.title(f'Architecture Latency Comparison for {transfer_description}')
    plt.savefig('architecture_comparison.png')
    print("✅ Graph 'architecture_comparison.png' has been saved.")

# --- --------------------------- ---
# --- MAIN SIMULATION EXECUTION ---
# --- --------------------------- ---
if __name__ == '__main__':
    # Define the network topology with (x, y) coordinates in km
    hub = "Central_EHR_Repository"
    spokes_config = {
        "Vellore_Main_Hospital": (10, 5),
        "Katpadi_Clinic": (-5, -5),
        "CMC_Diagnostic_Lab": (15, -10),
        "Ranipet_Branch": (-25, 20)
    }

    # Initialize the network object
    ehr_network = QKD_Network(hub_name=hub, hub_location=(0,0))
    for name, location in spokes_config.items():
        ehr_network.add_node(name, location)
        ehr_network.add_channel(hub, name)

    # --- DEMONSTRATION 1: ATTACK SIMULATION ---
    print("-" * 60)
    print("DEMONSTRATION 1: ATTACK SIMULATION")
    print("-" * 60)
    attacked_channel = tuple(sorted((hub, "Ranipet_Branch"))) # The longest link is most vulnerable
    ehr_network.establish_all_keys(protocol="BB84", attacked_link=attacked_channel)
    ehr_network.simulate_ehr_transfer("Vellore_Main_Hospital", "Ranipet_Branch", 512)

    # --- DEMONSTRATION 2: PROTOCOL COMPARISON ---
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: COMPARATIVE PROTOCOL ANALYSIS (BB84 vs B92)")
    print("=" * 60)
    ehr_network.key_store = {}
    ehr_network.establish_all_keys(protocol="BB84")
    bb84_results = {spoke: ehr_network.key_store.get((hub, spoke, "BB84"), 0) for spoke in spokes_config}
    ehr_network.key_store = {}
    ehr_network.establish_all_keys(protocol="B92")
    b92_results = {spoke: ehr_network.key_store.get((hub, spoke, "B92"), 0) for spoke in spokes_config}
    plot_protocol_comparison(bb84_results, b92_results)

    # --- DEMONSTRATION 3: ARCHITECTURE COMPARISON ---
    # --- DEMONSTRATION 3: ARCHITECTURE COMPARISON ---
print("\n" + "=" * 60)
print("DEMONSTRATION 3: COMPARATIVE ARCHITECTURE ANALYSIS (QKD vs PQC)")
print("=" * 60)

# Define the transfer we want to compare
source = "Katpadi_Clinic"
dest = "CMC_Diagnostic_Lab"

# 1. Get the latency for the QKD Network
qkd_result = ehr_network.simulate_ehr_transfer(source, dest, 512)
# FIX: Check if the result is None and replace with 0 if it is.
qkd_latency = qkd_result if qkd_result is not None else 0

# 2. Get the latency for a PQC Model
pqc_latency = simulate_pqc_ehr_transfer(ehr_network.nodes[source], ehr_network.nodes[dest], 512)

# 3. Plot the comparison
plot_architecture_comparison(qkd_latency, pqc_latency, f"{source} to {dest}")
qkd_latency = ehr_network.simulate_ehr_transfer(source, dest, 512)
pqc_latency = simulate_pqc_ehr_transfer(ehr_network.nodes[source], ehr_network.nodes[dest], 512)
plot_architecture_comparison(qkd_latency, pqc_latency, f"{source} to {dest}")

print("\nAll simulations complete. Displaying plots...")
plt.show()