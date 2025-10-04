import numpy as np
import time

# --- CONFIGURATION PARAMETERS ---
NUM_BITS_FOR_KEY_ESTABLISHMENT = 4096  # Number of qubits to generate keys
QBER_THRESHOLD = 0.05  # Abort protocol if QBER exceeds this (5%)
LATENCY_PER_KM_MS = 0.005  # Latency for light in fiber (approx. 5 microseconds/km)
HUB_PROCESSING_DELAY_MS = 15 # Fixed delay to simulate decryption/re-encryption at the hub

# --- CORE QKD COMPONENTS (from previous simulation) ---

class QuantumChannel:
    """Simulates the quantum channel with distance-based noise."""
    def __init__(self, distance, base_noise_level=0.01):
        self.distance = distance
        self.noise = base_noise_level + (distance / 1000)  # Noise increases with distance

    def transmit(self, qubits):
        """Transmits qubits, applying noise."""
        transmitted_qubits = []
        for bit, basis in qubits:
            if np.random.random() < self.noise:
                bit = 1 - bit  # Apply noise by flipping the bit
            transmitted_qubits.append((bit, basis))
        return transmitted_qubits

class Node:
    """Represents a healthcare entity in the QKD network."""
    def __init__(self, name):
        self.name = name
        self.bits = None
        self.bases = None

    def generate_qubits(self, num_bits):
        """Generates random bits and bases and prepares qubits for sending."""
        self.bits = np.random.randint(0, 2, num_bits)
        self.bases = np.random.randint(0, 2, num_bits)  # 0 for Rectilinear, 1 for Diagonal
        return list(zip(self.bits, self.bases))

    def measure_qubits(self, qubits):
        """Measures incoming qubits with new random bases."""
        self.bases = np.random.randint(0, 2, len(qubits))
        measured_bits = []
        for i, (received_bit, sender_basis) in enumerate(qubits):
            if self.bases[i] == sender_basis:
                measured_bits.append(received_bit)
            else:
                measured_bits.append(np.random.randint(0, 2))
        self.bits = np.array(measured_bits)

def run_bb84_link_simulation(node1, node2, channel, num_bits):
    """
    Orchestrates a single BB84 key establishment between two nodes.
    This function is now adapted to return a result dictionary.
    """
    # 1. Quantum Transmission
    qubits_sent = node1.generate_qubits(num_bits)
    qubits_received = channel.transmit(qubits_sent)
    node2.measure_qubits(qubits_received)

    # 2. Sifting
    matching_bases_indices = np.where(node1.bases == node2.bases)[0]
    sifted_key_node1 = node1.bits[matching_bases_indices]
    sifted_key_node2 = node2.bits[matching_bases_indices]

    # 3. QBER Estimation
    if len(sifted_key_node1) == 0:
        return {"success": False, "qber": 1, "final_key_length": 0}

    sample_size = min(len(sifted_key_node1) // 2, 500)
    sample_indices = np.random.choice(len(sifted_key_node1), sample_size, replace=False)
    
    sample1 = sifted_key_node1[sample_indices]
    sample2 = sifted_key_node2[sample_indices]
    errors = np.sum(sample1 != sample2)
    qber = errors / sample_size if sample_size > 0 else 0

    # 4. Check for success
    if qber >= QBER_THRESHOLD:
        return {"success": False, "qber": qber, "final_key_length": 0}

    # 5. Final Key Generation (Simulated)
    remaining_key_length = len(sifted_key_node1) - sample_size
    # Simulate privacy amplification by reducing key length based on QBER
    final_key_length = int(remaining_key_length * (1 - 2 * qber))
    
    return {"success": True, "qber": qber, "final_key_length": max(0, final_key_length)}

# --- NEW NETWORK SIMULATION CLASS ---

class QKD_Network:
    """Manages the entire trusted-node network, its topology, and simulations."""
    def __init__(self, hub_name):
        self.hub_name = hub_name
        self.nodes = {}
        self.channels = {}
        self.key_store = {}
        self.add_node(hub_name)

    def add_node(self, node_name):
        self.nodes[node_name] = Node(node_name)
        print(f"Node '{node_name}' added to the network.")

    def add_channel(self, node1_name, node2_name, distance):
        channel_name = tuple(sorted((node1_name, node2_name)))
        self.channels[channel_name] = QuantumChannel(distance=distance)
        print(f"Quantum channel added between '{node1_name}' and '{node2_name}' ({distance}km).")

    def establish_all_keys(self):
        """Runs BB84 between the hub and all other nodes to establish shared keys."""
        hub_node = self.nodes[self.hub_name]
        print("\n--- Establishing all network keys with the Hub ---")

        for spoke_name, spoke_node in self.nodes.items():
            if spoke_name == self.hub_name:
                continue

            channel_name = tuple(sorted((self.hub_name, spoke_name)))
            channel = self.channels[channel_name]
            
            result = run_bb84_link_simulation(hub_node, spoke_node, channel, NUM_BITS_FOR_KEY_ESTABLISHMENT)

            if result["success"]:
                key_len = result["final_key_length"]
                self.key_store[(self.hub_name, spoke_name)] = key_len
                self.key_store[(spoke_name, self.hub_name)] = key_len
                print(f"✅ SUCCESS: Key established between '{self.hub_name}' and '{spoke_name}'. Length: {key_len} bits (QBER: {result['qber']:.2%})")
            else:
                print(f"❌ FAILED: Key establishment failed between '{self.hub_name}' and '{spoke_name}'. (QBER: {result['qber']:.2%})")

    def simulate_ehr_transfer(self, source_name, dest_name, ehr_size_kb):
        """Simulates an EHR transfer from a source spoke to a destination spoke via the hub."""
        print(f"\n--- Simulating EHR transfer of {ehr_size_kb}KB from '{source_name}' to '{dest_name}' ---")

        if source_name == self.hub_name or dest_name == self.hub_name:
            print("Error: Source and destination must be spoke nodes.")
            return

        # Check for required keys
        if (source_name, self.hub_name) not in self.key_store or (self.hub_name, dest_name) not in self.key_store:
            print("Error: Missing a required secure key. Aborting transfer.")
            return

        total_latency = 0

        # Part 1: Source sends EHR to Hub
        channel_source_hub = self.channels[tuple(sorted((source_name, self.hub_name)))]
        latency1 = channel_source_hub.distance * LATENCY_PER_KM_MS
        total_latency += latency1
        print(f"1. Data transfer: {source_name} -> {self.hub_name} ({channel_source_hub.distance}km). Latency: {latency1:.2f} ms")

        # Part 2: Hub processes and re-encrypts the EHR
        total_latency += HUB_PROCESSING_DELAY_MS
        print(f"2. Processing at {self.hub_name}: Decryption/Re-encryption. Latency: {HUB_PROCESSING_DELAY_MS:.2f} ms")

        # Part 3: Hub sends EHR to Destination
        channel_hub_dest = self.channels[tuple(sorted((self.hub_name, dest_name)))]
        latency2 = channel_hub_dest.distance * LATENCY_PER_KM_MS
        total_latency += latency2
        print(f"3. Data transfer: {self.hub_name} -> {dest_name} ({channel_hub_dest.distance}km). Latency: {latency2:.2f} ms")

        print(f"--- ✅ COMPLETED: Total End-to-End Latency: {total_latency:.2f} ms ---")
        return total_latency

# --- MAIN SIMULATION SCRIPT ---
if __name__ == '__main__':
    # 1. Initialize the network with a central hub
    ehr_network = QKD_Network(hub_name="Central_EHR_Repository")

    # 2. Define the network topology
    spoke_nodes_config = {
        "Vellore_Main_Hospital": 15, # Node name and distance from hub in km
        "Katpadi_Clinic": 10,
        "CMC_Diagnostic_Lab": 25
    }
    for name, distance in spoke_nodes_config.items():
        ehr_network.add_node(name)
        ehr_network.add_channel(ehr_network.hub_name, name, distance)

    # 3. Establish all secure keys in the network
    ehr_network.establish_all_keys()
    
    # Check if keys were established before proceeding
    if not ehr_network.key_store:
        print("\nCould not establish any secure keys. Aborting transfer simulations.")
    else:
        # 4. Run transfer simulation scenarios
        ehr_network.simulate_ehr_transfer(
            source_name="Vellore_Main_Hospital",
            dest_name="CMC_Diagnostic_Lab",
            ehr_size_kb=512
        )
        
        ehr_network.simulate_ehr_transfer(
            source_name="Katpadi_Clinic",
            dest_name="Vellore_Main_Hospital",
            ehr_size_kb=128
        )