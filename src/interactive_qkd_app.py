import streamlit as st
import numpy as np
import pandas as pd
import time

# --- CORE BB84 SIMULATION LOGIC ---
# This is the heart of the QKD algorithm simulation.

def run_bb84_simulation(num_bits, noise_level, eve_is_present):
    """
    Performs a full BB84 simulation and returns a step-by-step log of the results.
    """
    results = {}

    # --- Stage 1: Alice prepares her qubits ---
    alice_bits = np.random.randint(0, 2, num_bits)
    alice_bases = np.random.randint(0, 2, num_bits) # 0 for +, 1 for x
    results['alice_bits'] = alice_bits
    results['alice_bases'] = alice_bases

    # --- Stage 2: Qubits travel through the channel (and maybe get intercepted by Eve) ---
    transmitted_qubits = []
    eve_bases = None
    if eve_is_present:
        eve_bases = np.random.randint(0, 2, num_bits)
        results['eve_bases'] = eve_bases

    for i in range(num_bits):
        bit = alice_bits[i]
        basis = alice_bases[i]
        
        # EVE'S ATTACK
        if eve_is_present:
            if eve_bases[i] != basis:
                bit = np.random.randint(0, 2) # Eve's wrong measurement randomizes the bit
        
        # CHANNEL NOISE 
        if np.random.random() < noise_level:
            bit = 1 - bit # Noise flips the bit
            
        transmitted_qubits.append(bit)

    # --- Stage 3: Bob measures the qubits ---
    bob_bases = np.random.randint(0, 2, num_bits)
    bob_results_list = []
    for i in range(num_bits):
        received_bit = transmitted_qubits[i]
        if alice_bases[i] == bob_bases[i]:
            bob_results_list.append(received_bit)
        else:
            bob_results_list.append(np.random.randint(0, 2)) # Random result if bases mismatch
    
    # --- FIX IS HERE: Convert bob_results to a NumPy array immediately ---
    bob_results = np.array(bob_results_list)

    results['bob_bases'] = bob_bases
    results['bob_results_full'] = bob_results
    
    # --- Stage 4: Sifting (Alice and Bob compare bases publicly) [cite: 79] ---
    matching_bases_indices = np.where(alice_bases == bob_bases)[0]
    
    alice_sifted_key = alice_bits[matching_bases_indices]
    bob_sifted_key = bob_results[matching_bases_indices] # This now works correctly
    
    results['matching_bases_indices'] = matching_bases_indices
    results['alice_sifted_key'] = alice_sifted_key
    results['bob_sifted_key'] = bob_sifted_key
    
    if len(alice_sifted_key) == 0:
        results['qber'] = 0
        results['final_key'] = ""
        results['attack_detected'] = False
        return results

    # --- Stage 5: Security Check (Calculate QBER)  ---
    # Alice and Bob sacrifice a portion of their key to check for errors.
    sample_size = len(alice_sifted_key) // 2
    sample_indices = np.random.choice(len(alice_sifted_key), sample_size, replace=False)
    
    errors = np.sum(alice_sifted_key[sample_indices] != bob_sifted_key[sample_indices])
    qber = errors / sample_size if sample_size > 0 else 0
    results['qber'] = qber
    
    # Check if QBER is above the security threshold
    QBER_THRESHOLD = 0.15 # Using a higher threshold for clear demonstration
    results['attack_detected'] = qber > QBER_THRESHOLD
    
    # --- Stage 6: Final Key Generation (Privacy Amplification) [cite: 79] ---
    if not results['attack_detected']:
        # Remove the sample bits used for QBER check
        final_key_indices = np.setdiff1d(np.arange(len(alice_sifted_key)), sample_indices)
        final_key_bits = alice_sifted_key[final_key_indices]
        results['final_key'] = "".join(map(str, final_key_bits))
    else:
        results['final_key'] = "KEY DISCARDED"
        
    return results


# --- STREAMLIT WEB APPLICATION UI ---

st.set_page_config(layout="wide")
st.title("🛡️ Interactive QKD Simulation for Secure EHRs")
st.markdown("""
This simulation demonstrates the **BB84 Quantum Key Distribution protocol** , a cornerstone of the conceptual framework for securing Electronic Health Records (EHRs)[cite: 2, 3].
Use the controls on the left to see how a secure key is established between two parties (Alice and Bob) and how the presence of an eavesdropper (Eve) is automatically detected.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
num_bits = st.sidebar.slider("Number of Qubits to Send", 10, 500, 100)
noise_level = st.sidebar.slider("Channel Noise Level", 0.0, 0.25, 0.02, 0.01, "%.2f")
eve_is_present = st.sidebar.checkbox("Enable Attacker (Eve)?")

if st.sidebar.button("Run Simulation"):
    
    with st.spinner("Simulating quantum exchange..."):
        time.sleep(1)
        results = run_bb84_simulation(num_bits, noise_level, eve_is_present)

    st.header("Simulation Results")
    
    # --- Display Summary Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Qubits Sent", num_bits)
    col2.metric("Sifted Key Length", len(results['alice_sifted_key']))
    col3.metric("Final Secure Key Length", len(results['final_key']) if not results['attack_detected'] else "N/A")
    col4.metric("Quantum Bit Error Rate (QBER)", f"{results['qber']:.2%}")

    # --- Display Final Security Status ---
    if results['attack_detected']:
        st.error(f"**SECURITY ALERT!** QBER of {results['qber']:.2%} is above the threshold. Attack detected! The key has been discarded.")
    else:
        st.success(f"**SECURE KEY ESTABLISHED!** QBER of {results['qber']:.2%} is within safe limits.")

    st.info(f"**Final Secure Key:** `{results['final_key']}`")

    # --- Display Detailed Step-by-Step Breakdown ---
    st.header("Step-by-Step Protocol Breakdown")

    # Create a DataFrame for visualization
    df_data = {
        'Alice\'s Bit': results['alice_bits'],
        'Alice\'s Basis (+/x)': ['+' if b == 0 else 'x' for b in results['alice_bases']],
        'Bob\'s Basis (+/x)': ['+' if b == 0 else 'x' for b in results['bob_bases']],
    }
    if eve_is_present:
        df_data['Eve\'s Basis (+/x)'] = ['+' if b == 0 else 'x' for b in results['eve_bases']]
    
    df = pd.DataFrame(df_data)
    
    df['Bases Match?'] = (results['alice_bases'] == results['bob_bases'])
    
    with st.expander("Show Detailed Qubit Exchange Log", expanded=False):
        st.dataframe(df)

    st.markdown("""
    **How to read the log:**
    1.  **Alice's Bit & Basis:** Alice randomly chooses a bit (0 or 1) and a basis ('+' or 'x') to encode it.
    2.  **Eve's Basis (if present):** The attacker, Eve, intercepts and measures using her own random basis. If her basis doesn't match Alice's, she corrupts the qubit's state.
    3.  **Bob's Basis:** Bob measures the incoming qubit using his own random basis.
    4.  **Bases Match?:** After the transmission, Alice and Bob publicly announce their bases. They **only keep the bits where their bases matched** (this is the sifting process)[cite: 79].
    5.  **QBER Calculation:** They then compare a small, random sample of their sifted keys. Any disagreements are counted as errors. A high error rate (QBER) implies an eavesdropper was present.
    """)