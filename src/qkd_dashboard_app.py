import streamlit as st
import numpy as np
import time
import math

# --- CORE SIMULATION LOGIC (BB84 Protocol) ---
# This part contains the physics and rules of the QKD exchange.

QBER_THRESHOLD = 0.10 # A bit higher for clearer demonstration

def run_bb84_simulation(eavesdropper_present=False):
    """Simulates a single BB84 exchange and returns the QBER."""
    num_bits = 1024 # A fixed number for this simulation
    alice_bits = np.random.randint(0, 2, num_bits)
    alice_bases = np.random.randint(0, 2, num_bits)
    
    # Simulate transmission with potential eavesdropper
    bob_results = []
    for i in range(num_bits):
        bit = alice_bits[i]
        basis = alice_bases[i]
        
        if eavesdropper_present:
            eve_basis = np.random.randint(0, 2)
            if basis != eve_basis: bit = np.random.randint(0, 2)

        bob_basis = np.random.randint(0, 2)
        if basis == bob_basis:
            bob_results.append((i, bit)) # Bob gets the (potentially altered) bit
            
    # Sifting and QBER Calculation
    if len(bob_results) < 2: return 1.0 # Failed
    
    sifted_indices = [res[0] for res in bob_results]
    sifted_bob_bits = np.array([res[1] for res in bob_results])
    sifted_alice_bits = alice_bits[sifted_indices]
    
    sample_size = len(sifted_alice_bits) // 2
    if sample_size == 0: return 1.0 # Failed
    
    sample_indices = np.random.choice(len(sifted_alice_bits), sample_size, replace=False)
    errors = np.sum(sifted_alice_bits[sample_indices] != sifted_bob_bits[sample_indices])
    qber = errors / sample_size
    return qber

# --- STREAMLIT APPLICATION ---

st.set_page_config(layout="wide", page_title="QKD Network Dashboard")

# Initialize the session state to store network status
if 'network_status' not in st.session_state:
    st.session_state.network_status = {}
if 'log' not in st.session_state:
    st.session_state.log = ["Welcome to the QKD Network Operations Center."]

def log_event(message):
    """Adds a message to the event log."""
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

# --- UI LAYOUT ---

st.title("🛡️ QKD Secure Healthcare Network Dashboard")
st.markdown("Simulating the real-time operation and security of a metropolitan EHR network.")

# Define the network topology
hub_name = "Central Repository"
hospital_names = ["Vellore Hospital", "CMC Lab", "Katpadi Clinic"]

# --- Main Dashboard Area ---
st.subheader("Live Network Status")
cols = st.columns(len(hospital_names))
for i, name in enumerate(hospital_names):
    status = st.session_state.network_status.get(name, {"status": "Offline", "qber": 0})
    with cols[i]:
        if status['status'] == 'Secure':
            st.success(f"**{name}**\n\nStatus: {status['status']}\n\nQBER: {status['qber']:.2%}")
        elif status['status'] == 'Compromised':
            st.error(f"**{name}**\n\nStatus: {status['status']}\n\nQBER: {status['qber']:.2%}")
        else:
            st.info(f"**{name}**\n\nStatus: {status['status']}")

st.markdown("---")

# --- Operator Actions Area ---
st.subheader("Operator Actions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**1. Secure the Network**")
    attack_target = st.selectbox("Select a link to attack (or None):", ["None"] + hospital_names)
    if st.button("Initiate Quantum Key Exchange"):
        log_event("Starting network-wide key establishment...")
        with st.spinner("Running BB84 protocol for all links..."):
            for name in hospital_names:
                is_attacked = (name == attack_target)
                qber = run_bb84_simulation(eavesdropper_present=is_attacked)
                if qber > QBER_THRESHOLD:
                    st.session_state.network_status[name] = {"status": "Compromised", "qber": qber}
                    log_event(f"🔴 SECURITY ALERT: High QBER on '{name}' link! Key discarded.")
                else:
                    st.session_state.network_status[name] = {"status": "Secure", "qber": qber}
                    log_event(f"🟢 SECURE: Key established for '{name}'.")
        log_event("Key establishment process complete.")
        st.rerun()

with col2:
    st.markdown("**2. Simulate EHR Data Fetch**")
    source_hosp = st.selectbox("Doctor at:", hospital_names, key='src')
    dest_hosp = st.selectbox("Requests data from:", hospital_names, key='dst')
    if st.button("Fetch Patient Record"):
        if source_hosp == dest_hosp:
            log_event("⚠️ INFO: Source and destination cannot be the same.")
        else:
            log_event(f"Request initiated: '{source_hosp}' -> '{dest_hosp}'.")
            source_status = st.session_state.network_status.get(source_hosp, {}).get('status')
            dest_status = st.session_state.network_status.get(dest_hosp, {}).get('status')
            
            if source_status == 'Secure' and dest_status == 'Secure':
                log_event(f"✅ SUCCESS: Keys are secure. Data transfer is authorized and proceeds.")
            else:
                log_event(f"❌ FAILED: Transfer aborted. Required link is not secure.")
                if source_status != 'Secure': log_event(f"Reason: Link to '{source_hosp}' is {source_status or 'Offline'}.")
                if dest_status != 'Secure': log_event(f"Reason: Link to '{dest_hosp}' is {dest_status or 'Offline'}.")

# --- Event Log Area ---
st.markdown("---")
st.subheader("Event Log")
st.text_area("", value="\n".join(st.session_state.log), height=300, key="log_area")