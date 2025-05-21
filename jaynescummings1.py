import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, expm

def create_basis_vector(state_idx, dim):
    """
    Create a basis vector |state_idx⟩ in a space of dimension dim.
    """
    vec = np.zeros(dim, dtype=complex)
    vec[state_idx] = 1.0
    return vec

def tensor_product(state1, state2):
    """
    Compute the tensor product of two state vectors.
    """
    return np.kron(state1, state2)

def von_neumann_entropy(density_matrix):
    """
    Calculate the von Neumann entropy of a density matrix.
    S(ρ) = -Tr(ρ log₂ ρ) = -∑ λᵢ log₂ λᵢ
    
    Includes handling for numerical stability.
    """
    # Ensure density matrix is Hermitian (it should be, but numerical errors can occur)
    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)
    
    # Get eigenvalues 
    eigenvalues = np.real(eigh(density_matrix, eigvals_only=True))
    
    # Normalize eigenvalues to ensure they sum to 1
    total = np.sum(eigenvalues)
    if abs(total - 1.0) > 1e-10:  # If sum is not close to 1
        eigenvalues = eigenvalues / total
    
    # Filter out very small eigenvalues (numerical errors)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    # Calculate entropy: -∑ λᵢ log₂ λᵢ
    # Add a small value inside log to avoid log(0)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
    
    # Ensure non-negative result (entropy can never be negative)
    return max(0.0, entropy)

def partial_trace_B(rho_AB, dim_A, dim_B):
    """
    Compute the partial trace over system B, returning reduced density matrix of A.
    Includes normalization and Hermiticity enforcement.
    """
    rho_A = np.zeros((dim_A, dim_A), dtype=complex)
    
    for i in range(dim_A):
        for j in range(dim_A):
            for k in range(dim_B):
                rho_A[i, j] += rho_AB[i * dim_B + k, j * dim_B + k]
    
    # Ensure Hermiticity to avoid numerical errors
    rho_A = 0.5 * (rho_A + rho_A.conj().T)
    
    # Normalize to ensure trace = 1
    trace = np.trace(rho_A)
    if abs(trace - 1.0) > 1e-10:  # If trace is not close to 1
        rho_A = rho_A / trace
        
    return rho_A

def partial_trace_A(rho_AB, dim_A, dim_B):
    """
    Compute the partial trace over system A, returning reduced density matrix of B.
    Includes normalization and Hermiticity enforcement.
    """
    rho_B = np.zeros((dim_B, dim_B), dtype=complex)
    
    for i in range(dim_B):
        for j in range(dim_B):
            for k in range(dim_A):
                rho_B[i, j] += rho_AB[k * dim_B + i, k * dim_B + j]
    
    # Ensure Hermiticity to avoid numerical errors
    rho_B = 0.5 * (rho_B + rho_B.conj().T)
    
    # Normalize to ensure trace = 1
    trace = np.trace(rho_B)
    if abs(trace - 1.0) > 1e-10:  # If trace is not close to 1
        rho_B = rho_B / trace
        
    return rho_B

def jaynes_cummings_hamiltonian(dim_field, g=1.0):
    """
    Create the Jaynes-Cummings Hamiltonian in the rotating wave approximation.
    H = g(a†σ- + aσ+)
    
    Returns a matrix representing the interaction Hamiltonian for a single atom.
    
    Args:
        dim_field: Dimension of the field Hilbert space
        g: Coupling constant (default: 1.0)
    """
    # Dimensions of the system
    dim_atom = 2
    dim_total = dim_atom * dim_field
    
    # Initialize Hamiltonian
    H = np.zeros((dim_total, dim_total), dtype=complex)
    
    # Loop through all possible field excitations
    for n in range(dim_field - 1):  # n goes from 0 to dim_field-2
        # |1⟩|n⟩ -> |0⟩|n+1⟩: a†σ- term
        idx_1n = 1 * dim_field + n
        idx_0np1 = 0 * dim_field + (n + 1)
        
        # Coupling strength with √(n+1) factor from field operator
        coupling = g * np.sqrt(n + 1)
        
        # Add coupling terms (both directions due to Hermiticity)
        H[idx_0np1, idx_1n] = coupling  # a†σ-
        H[idx_1n, idx_0np1] = coupling  # aσ+
    
    return H

def time_evolution_operator(H, time):
    """
    Calculate the time evolution operator U = exp(-i*H*t).
    We set ħ = 1 for simplicity.
    
    Uses scipy's expm for more stable matrix exponential calculation.
    """
    # Use scipy's expm which is more stable than np.exp for matrix exponentials
    return expm(-1j * H * time)

def fixed_interaction_simulation(M, N, g=1.0):
    """
    Case 1: Simulate with fixed interaction time for all atoms and all field states.
    
    Args:
        M: Number of atoms
        N: Initial field excitation number
        g: Coupling constant (default: 1.0)
        
    Returns:
        entropies_B: List of entanglement entropies of subsystem B after each atom interaction
    """
    # Dimension of atom space (2 states: ground |0⟩ and excited |1⟩)
    dim_atom = 2
    
    # Dimension of field space (need to account for states |0⟩ through |N⟩)
    dim_field = N + 1
    
    # Create Hamiltonian
    H = jaynes_cummings_hamiltonian(dim_field, g)
    
    # Calculate the required interaction time
    if N > 0:
        fixed_time = np.pi/(4 * g * np.sqrt(N))
    else:
        fixed_time = np.pi/(4 * g)  # Use a non-zero time for N=0
    
    # Generate the time evolution operator with this fixed time
    U_fixed = time_evolution_operator(H, fixed_time)
    
    # Initialize field state |N⟩
    field_state = create_basis_vector(N, dim_field)
    field_dm = np.outer(field_state, np.conj(field_state))
    
    # Ensure field density matrix is properly formed
    field_dm = 0.5 * (field_dm + field_dm.conj().T)  # Ensure Hermiticity
    
    # Array to store entropy values - start with initial entropy (0 for pure state)
    entropies_B = [0.0]  # Initial entropy of pure state |N⟩ is 0
    
    # Create the number operator (diagonal matrix with 0, 1, 2, ..., N)
    number_operator = np.diag(np.arange(dim_field))
    
    # Track average excitation number
    avg_excitations = [N]  # Initial average excitation is N
    
    # Starting with field state density matrix
    rho_B = field_dm
    
    # Iterate through M atoms
    for atom_idx in range(M):
        # Calculate average excitation number (for tracking)
        avg_n = np.real(np.trace(rho_B @ number_operator))
        avg_excitations.append(avg_n)
        
        # Initialize atom in ground state |0⟩
        atom_state = create_basis_vector(0, dim_atom)
        atom_dm = np.outer(atom_state, np.conj(atom_state))
        
        # Ensure atom density matrix is properly formed
        atom_dm = 0.5 * (atom_dm + atom_dm.conj().T)  # Ensure Hermiticity
        
        # Tensor product to get combined system state before interaction
        rho_combined = np.kron(atom_dm, rho_B)
        
        # Apply the fixed time evolution operator
        rho_after = U_fixed @ rho_combined @ U_fixed.conj().T
        
        # Ensure the combined density matrix remains Hermitian
        rho_after = 0.5 * (rho_after + rho_after.conj().T)
        
        # Compute reduced density matrix for field (subsystem B)
        rho_B = partial_trace_A(rho_after, dim_atom, dim_field)
        
        # Calculate entropy of subsystem B
        entropy_B = von_neumann_entropy(rho_B)
        entropies_B.append(entropy_B)
    
    return entropies_B, avg_excitations

def variable_interaction_simulation(M, N, g=1.0):
    """
    Case 2: Simulate with variable interaction time for each atom based on average field excitation.
    
    Interaction time for each atom is determined by the average excitation number
    of the field at the beginning of each interaction: t_k = π/(4g*√n_avg,k)
    
    This implementation allows for M > N by using the field's average excitation.
    
    Args:
        M: Number of atoms (can be greater than N+1)
        N: Initial field excitation number
        g: Coupling constant (default: 1.0)
        
    Returns:
        entropies_B: List of entanglement entropies of subsystem B after each atom interaction
        avg_excitations: List of average excitation numbers used for time calculation
    """
    # Dimension of atom space
    dim_atom = 2
    
    # Dimension of field space (need to account for states |0⟩ through |N⟩)
    dim_field = N + 1
    
    # Create Hamiltonian
    H = jaynes_cummings_hamiltonian(dim_field, g)
    
    # Initialize field state |N⟩
    field_state = create_basis_vector(N, dim_field)
    field_dm = np.outer(field_state, np.conj(field_state))
    
    # Ensure field density matrix is properly formed
    field_dm = 0.5 * (field_dm + field_dm.conj().T)  # Ensure Hermiticity
    
    # Array to store entropy values - start with initial entropy (0 for pure state)
    entropies_B = [0.0]  # Initial entropy of pure state |N⟩ is 0
    
    # Create the number operator (diagonal matrix with 0, 1, 2, ..., N)
    number_operator = np.diag(np.arange(dim_field))
    
    # Array to store average excitation numbers
    avg_excitations = [N]  # Initial average excitation is N
    
    # Starting with field state density matrix
    rho_B = field_dm
    
    # Iterate through M atoms
    for k in range(1, M+1):
        # Calculate average excitation number of the field: <n> = Tr(ρ_B * n)
        avg_n = np.real(np.trace(rho_B @ number_operator))
        avg_excitations.append(avg_n)
        
        # Calculate interaction time based on average excitation
        if avg_n > 0:
            # t_k = π/(4g*√n_avg,k)
            variable_time = np.pi / (4 * g * np.sqrt(avg_n))
        else:
            # No excitations left, use minimal time
            variable_time = np.pi / (4 * g)  # Use a small non-zero time
        
        # Generate the time evolution operator for this specific atom
        U_variable = time_evolution_operator(H, variable_time)
        
        # Initialize atom in ground state |0⟩
        atom_state = create_basis_vector(0, dim_atom)
        atom_dm = np.outer(atom_state, np.conj(atom_state))
        
        # Ensure atom density matrix is properly formed
        atom_dm = 0.5 * (atom_dm + atom_dm.conj().T)
        
        # Tensor product to get combined system state before interaction
        rho_combined = np.kron(atom_dm, rho_B)
        
        # Apply the variable time evolution operator
        rho_after = U_variable @ rho_combined @ U_variable.conj().T
        
        # Ensure the combined density matrix remains Hermitian
        rho_after = 0.5 * (rho_after + rho_after.conj().T)
        
        # Compute reduced density matrix for field (subsystem B)
        rho_B = partial_trace_A(rho_after, dim_atom, dim_field)
        
        # Calculate entropy of subsystem B
        entropy_B = von_neumann_entropy(rho_B)
        entropies_B.append(entropy_B)
    
    return entropies_B, avg_excitations

def plot_entropy(entropies, M, N, title_suffix=""):
    """
    Plot the entanglement entropy of subsystem B over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, M+1), entropies, 'bo-', linewidth=2)
    plt.xlabel('Number of Atoms Interacted', fontsize=12)
    plt.ylabel('Entanglement Entropy of Field (Subsystem B)', fontsize=12)
    plt.title(f'Entanglement Entropy Evolution in Modified Jaynes-Cummings Model\nInitial Field State: |{N}⟩, M={M} Atoms{title_suffix}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Handle x-axis tick labels adaptively to prevent overlap
    if M <= 15:
        # For small M, show all ticks
        plt.xticks(range(0, M+1))
    elif M <= 50:
        # For medium M, show every 5th tick
        step = 5
        ticks = list(range(0, M+1, step))
        if M % step != 0:  # Make sure the last tick is included
            ticks.append(M)
        plt.xticks(ticks)
    else:
        # For large M, determine step size dynamically
        step = max(1, M // 10)  # Show about 10 ticks
        ticks = list(range(0, M+1, step))
        if M % step != 0:  # Make sure the last tick is included
            ticks.append(M)
        plt.xticks(ticks)
    
    plt.ylim(bottom=0)  # Start y-axis at 0
    plt.tight_layout()
    return plt

def plot_excitation(avg_excitations, M, N, title_suffix=""):
    """
    Plot the average excitation number of field over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, M+1), avg_excitations, 'go-', linewidth=2)
    plt.xlabel('Number of Atoms Interacted', fontsize=12)
    plt.ylabel('Average Excitation Number ⟨n⟩', fontsize=12)
    plt.title(f'Average Field Excitation Evolution in Modified Jaynes-Cummings Model\nInitial Field State: |{N}⟩, M={M} Atoms{title_suffix}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Handle x-axis tick labels adaptively
    if M <= 15:
        plt.xticks(range(0, M+1))
    elif M <= 50:
        step = 5
        ticks = list(range(0, M+1, step))
        if M % step != 0:
            ticks.append(M)
        plt.xticks(ticks)
    else:
        step = max(1, M // 10)
        ticks = list(range(0, M+1, step))
        if M % step != 0:
            ticks.append(M)
        plt.xticks(ticks)
    
    plt.ylim(bottom=0)  # Start y-axis at 0
    plt.tight_layout()
    return plt

def plot_comparison(entropies1, entropies2, M, N):
    """
    Create a plot comparing the two simulation methods.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, M+1), entropies1, 'bo-', linewidth=2, label="Case 1: Fixed Time")
    plt.plot(range(0, M+1), entropies2, 'ro-', linewidth=2, label="Case 2: Variable Time")
    plt.xlabel('Number of Atoms Interacted', fontsize=12)
    plt.ylabel('Entanglement Entropy of Field (Subsystem B)', fontsize=12)
    plt.title(f'Comparison of Entanglement Entropy Evolution\nInitial Field State: |{N}⟩, M={M} Atoms', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Handle x-axis tick labels adaptively
    if M <= 15:
        plt.xticks(range(0, M+1))
    elif M <= 50:
        step = 5
        ticks = list(range(0, M+1, step))
        if M % step != 0:
            ticks.append(M)
        plt.xticks(ticks)
    else:
        step = max(1, M // 10)
        ticks = list(range(0, M+1, step))
        if M % step != 0:
            ticks.append(M)
        plt.xticks(ticks)
    
    plt.ylim(bottom=0)
    plt.tight_layout()
    return plt

def plot_excitation_comparison(avg_exc1, avg_exc2, M, N):
    """
    Create a plot comparing the average excitation evolution between the two methods.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, M+1), avg_exc1, 'bo-', linewidth=2, label="Case 1: Fixed Time")
    plt.plot(range(0, M+1), avg_exc2, 'ro-', linewidth=2, label="Case 2: Variable Time")
    plt.xlabel('Number of Atoms Interacted', fontsize=12)
    plt.ylabel('Average Excitation Number ⟨n⟩', fontsize=12)
    plt.title(f'Comparison of Average Field Excitation Evolution\nInitial Field State: |{N}⟩, M={M} Atoms', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Handle x-axis tick labels adaptively
    if M <= 15:
        plt.xticks(range(0, M+1))
    elif M <= 50:
        step = 5
        ticks = list(range(0, M+1, step))
        if M % step != 0:
            ticks.append(M)
        plt.xticks(ticks)
    else:
        step = max(1, M // 10)
        ticks = list(range(0, M+1, step))
        if M % step != 0:
            ticks.append(M)
        plt.xticks(ticks)
    
    plt.ylim(bottom=0)
    plt.tight_layout()
    return plt

# Parameters
M = 40  # Number of atoms (now M can be > N)
N = 10  # Initial field excitation number 
g = 1.0  # Coupling constant

# Run simulations with the updated methods
print("Running Case 1: Fixed interaction time...")
entropies_case1, avg_exc1 = fixed_interaction_simulation(M, N, g)

print("Running Case 2: Variable interaction time based on average excitation...")
entropies_case2, avg_exc2 = variable_interaction_simulation(M, N, g)

# Plot individual entropy results
plt1 = plot_entropy(entropies_case1, M, N, title_suffix=" (Fixed Interaction Time)")
plt1.savefig('entropy_fixed_time.png')
plt1.show()

plt2 = plot_entropy(entropies_case2, M, N, title_suffix=" (Variable Interaction Time)")
plt2.savefig('entropy_variable_time.png')
plt2.show()

# Plot comparisons
plt_comp1 = plot_comparison(entropies_case1, entropies_case2, M, N)
plt_comp1.savefig('entropy_comparison.png')
plt_comp1.show()
