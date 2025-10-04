import numpy as np
import time
import matplotlib.pyplot as plt
import itertools

# --- --------------------------------- ---
# --- GLOBAL SIMULATION CONFIGURATION ---
# --- --------------------------------- ---
NUM_BITS_FOR_KEY_ESTABLISHMENT = 4096
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
        """Transmits qubits, applying noise and simulating an eavesdropper ('Eve')."""
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
            if self.bases[i] == sender_basis:
                measured_bits.append(received_bit)
            else:
                measured_bits.append(np.random.randint(0, 2))
        self.bits = np.array(measured_bits)

# --- ------------------------------------- ---
# --- BB84 LINK SIMULATION & NETWORK CLASS ---
# --- ------------------------------------- ---

def run_bb84_link_simulation(sender, receiver, channel, num_bits, eavesdropper_present=False):
    qubits_sent = sender.prepare_qubits(num_bits)
    qubits_received = channel.transmit(qubits_sent, eavesdropper_present=eavesdropper_present)
    receiver.measure_qubits(qubits_received)
    matching_bases_indices = np.where(sender.bases == receiver.bases)[0]
    sifted_key_sender = sender.bits[matching_bases_indices]
    sifted_key_receiver = receiver.bits[matching_bases_indices]
    if len(sifted_key_sender) < 20: # Ensure sample size is reasonable
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

class QKD_Network:
    """Manages the trusted-node network, its topology, and all simulations."""
    def __init__(self, hub_name):
        self.hub_name = hub_name
        self.nodes = {}
        self.channels = {}
        self.key_store = {}
        self.add_node(hub_name)

    def add_node(self, node_name):
        self.nodes[node_name] = Node(node_name)

    def add_channel(self, node1_name, node2_name, distance):
        channel_name = tuple(sorted((node1_name, node2_name)))
        self.channels[channel_name] = QuantumChannel(distance=distance)

    def establish_all_keys(self, attacked_link=None):
        hub_node = self.nodes[self.hub_name]
        print(f"\n--- Establishing Keys for the Network (Attacked Link: {attacked_link or 'None'}) ---")
        for spoke_name in self.nodes:
            if spoke_name == self.hub_name: continue
            is_attacked = tuple(sorted((self.hub_name, spoke_name))) == attacked_link
            spoke_node = self.nodes[spoke_name]
            channel = self.channels[tuple(sorted((self.hub_name, spoke_name)))]
            result = run_bb84_link_simulation(hub_node, spoke_node, channel, NUM_BITS_FOR_KEY_ESTABLISHMENT, eavesdropper_present=is_attacked)
            if result["success"]:
                key_len = result["final_key_length"]
                self.key_store[(self.hub_name, spoke_name)] = key_len
                print(f"✅ SUCCESS: Key between '{self.hub_name}' and '{spoke_name}'. Length: {key_len} bits (QBER: {result['qber']:.2%})")
            else:
                print(f"❌ FAILED: Key establishment failed for '{spoke_name}'. (QBER: {result['qber']:.2%}) - ATTACK DETECTED!")

    def simulate_ehr_transfer(self, source_name, dest_name, ehr_size_kb):
        print(f"\n--- Attempting EHR transfer of {ehr_size_kb}KB from '{source_name}' to '{dest_name}' ---")
        if (self.hub_name, source_name) not in self.key_store or (self.hub_name, dest_name) not in self.key_store:
            print("❌ ABORTED: Cannot perform transfer. A required secure key is missing.")
            return None
        channel1 = self.channels[tuple(sorted((source_name, self.hub_name)))]
        latency1 = channel1.distance * LATENCY_PER_KM_MS
        channel2 = self.channels[tuple(sorted((dest_name, self.hub_name)))]
        latency2 = channel2.distance * LATENCY_PER_KM_MS
        total_latency = latency1 + HUB_PROCESSING_DELAY_MS + latency2
        print(f"1. Data transfer {source_name} -> {self.hub_name} ({channel1.distance}km). Time: {latency1:.2f} ms")
        print(f"2. Processing at {self.hub_name}. Time: {HUB_PROCESSING_DELAY_MS:.2f} ms")
        print(f"3. Data transfer {self.hub_name} -> {dest_name} ({channel2.distance}km). Time: {latency2:.2f} ms")
        print(f"--- ✅ COMPLETED: Total End-to-End Latency: {total_latency:.2f} ms ---")
        
        # Return a dictionary with detailed results for graphing
        return {
            "total_latency": total_latency,
            "latency_source_hub": latency1,
            "latency_hub_processing": HUB_PROCESSING_DELAY_MS,
            "latency_hub_dest": latency2,
            "total_distance": channel1.distance + channel2.distance
        }

# --- --------------------- ---
# --- NEW GRAPHING FUNCTIONS ---
# --- --------------------- ---

def generate_latency_vs_distance_plot(results_list):
    """Generates a scatter plot of total latency vs. total path distance."""
    if not results_list:
        print("No data available to generate latency vs. distance plot.")
        return
        
    distances = [r['total_distance'] for r in results_list]
    latencies = [r['total_latency'] for r in results_list]
    labels = [r['label'] for r in results_list]

    plt.figure(figsize=(10, 6))
    plt.scatter(distances, latencies, c='blue')

    # Add labels to points
    for i, label in enumerate(labels):
        plt.annotate(label, (distances[i], latencies[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('End-to-End Latency vs. Total Path Distance in Trusted-Node Network')
    plt.xlabel('Total Path Distance (Source -> Hub -> Destination) in km')
    plt.ylabel('Total End-to-End Latency (ms)')
    plt.grid(True)
    plt.savefig('latency_vs_distance.png')
    print("\n✅ Graph 'latency_vs_distance.png' has been saved.")

def generate_latency_breakdown_plot(result, source_name, dest_name):
    """Generates a bar chart showing the breakdown of latency for a single transfer."""
    if not result:
        print(f"No data to generate breakdown plot for {source_name} -> {dest_name}.")
        return

    labels = [f'{source_name} to Hub', 'Hub Processing', f'Hub to {dest_name}']
    latencies = [result['latency_source_hub'], result['latency_hub_processing'], result['latency_hub_dest']]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, latencies, color=['skyblue', 'salmon', 'lightgreen'])
    
    # Add text labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f} ms', va='bottom', ha='center')

    plt.title(f'Latency Breakdown for Transfer: {source_name} to {dest_name}')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('latency_breakdown.png')
    print("✅ Graph 'latency_breakdown.png' has been saved.")

# --- --------------------------- ---
# --- MAIN SIMULATION EXECUTION ---
# --- --------------------------- ---
if __name__ == '__main__':
    # Define the network topology
    hub = "Central_EHR_Repository"
    spokes = { "Vellore_Main_Hospital": 15, "Katpadi_Clinic": 10, "CMC_Diagnostic_Lab": 25 }

    # Initialize the network
    ehr_network = QKD_Network(hub_name=hub)
    for spoke_name, distance in spokes.items():
        ehr_network.add_node(spoke_name)
        ehr_network.add_channel(hub, spoke_name, distance)

    print("-" * 60)
    print("SCENARIO A: NORMAL OPERATION (NO ATTACKS)")
    print("-" * 60)
    ehr_network.establish_all_keys(attacked_link=None)
    
    # Run all possible transfers to gather data for the first plot
    all_transfer_results = []
    spoke_names = list(spokes.keys())
    for source, dest in itertools.permutations(spoke_names, 2):
        result = ehr_network.simulate_ehr_transfer(source, dest, 256)
        if result:
            result['label'] = f'{source[:1]}->{dest[:1]}' # Short label for plot
            all_transfer_results.append(result)
            
    # Generate the latency vs. distance plot from all transfers
    generate_latency_vs_distance_plot(all_transfer_results)
    
    # Generate a breakdown plot for one specific, interesting transfer
    # Let's pick the longest one: Vellore_Main_Hospital -> CMC_Diagnostic_Lab
    specific_transfer_result = ehr_network.simulate_ehr_transfer("Vellore_Main_Hospital", "CMC_Diagnostic_Lab", 512)
    generate_latency_breakdown_plot(specific_transfer_result, "Vellore_Main_Hospital", "CMC_Diagnostic_Lab")

    print("\n" + "=" * 60)
    print("SCENARIO B: ATTACK ON ONE CHANNEL")
    print("=" * 60)
    ehr_network.key_store = {} # Reset keys
    attacked_channel = tuple(sorted((hub, "Vellore_Main_Hospital")))
    ehr_network.establish_all_keys(attacked_link=attacked_channel)
    ehr_network.simulate_ehr_transfer("Vellore_Main_Hospital", "CMC_Diagnostic_Lab", 512)
    
    # Show the generated plots
    plt.show()