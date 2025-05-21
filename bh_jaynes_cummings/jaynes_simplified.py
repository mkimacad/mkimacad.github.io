import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

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


def simulate_modified_jaynes_cummings(M, N):
    """
    Simulate the modified Jaynes-Cummings model with M atoms and initial field state |N⟩

    Args:
        M: Number of atoms
        N: Initial field excitation number

    Returns:
        entropies_B: List of entanglement entropies of subsystem B after each atom interaction
    """
    # Dimension of atom space (2 states: ground |0⟩ and excited |1⟩)
    dim_atom = 2

    # Dimension of field space (need to account for states |0⟩ through |N⟩)
    dim_field = N + 1

    # Initialize field state |N⟩
    field_state = create_basis_vector(N, dim_field)
    field_dm = np.outer(field_state, np.conj(field_state))

    # Array to store entropy values - start with initial entropy (0 for pure state)
    entropies_B = [0.0]  # Initial entropy of pure state |N⟩ is 0

    # Starting with field state density matrix
    rho_B = field_dm

    # Iterate through M atoms
    for atom_idx in range(M):
        # Initialize atom in ground state |0⟩
        atom_state = create_basis_vector(0, dim_atom)
        atom_dm = np.outer(atom_state, np.conj(atom_state))

        # Tensor product to get combined system state before interaction
        rho_combined = np.kron(atom_dm, rho_B)

        # Apply the interaction unitary: U|0_A⟩|n⟩ = 1/√2(|0_A⟩|n⟩+|1_A⟩|n-1⟩)

        # Create the interaction unitary matrix
        U = np.zeros((dim_atom * dim_field, dim_atom * dim_field), dtype=complex)

        # Loop through all possible field states
        for n in range(dim_field):
            # Index for |0⟩|n⟩
            idx_0n = 0 * dim_field + n

            # For |0⟩|n⟩ -> 1/√2(|0⟩|n⟩ + |1⟩|n-1⟩)
            U[idx_0n, idx_0n] = 1/np.sqrt(2)  # |0⟩|n⟩ -> |0⟩|n⟩ component

            if n > 0:  # Only add |1⟩|n-1⟩ term if n > 0
                idx_1nm1 = 1 * dim_field + (n-1)  # Index for |1⟩|n-1⟩
                U[idx_1nm1, idx_0n] = 1/np.sqrt(2)  # |0⟩|n⟩ -> |1⟩|n-1⟩ component

            # Keep |1⟩|n⟩ states unchanged
            if n < dim_field - 1:  # Only applicable for valid n values
                idx_1n = 1 * dim_field + n  # Index for |1⟩|n⟩
                U[idx_1n, idx_1n] = 1

        # Apply unitary transformation: ρ' = U ρ U†
        rho_after = U @ rho_combined @ U.conj().T

        # Compute reduced density matrix for field (subsystem B)
        rho_B = partial_trace_A(rho_after, dim_atom, dim_field)

        # Calculate entropy of subsystem B
        entropy_B = von_neumann_entropy(rho_B)
        entropies_B.append(entropy_B)

    return entropies_B

def plot_entropy(entropies, M, N):
    """
    Plot the entanglement entropy of subsystem B over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, M+1), entropies, 'bo-', linewidth=2)
    plt.xlabel('Number of Atoms Interacted', fontsize=12)
    plt.ylabel('Entanglement Entropy of Field (Subsystem B)', fontsize=12)
    plt.title(f'Entanglement Entropy Evolution in simplified Jaynes-Cummings\nInitial Field State: |{N}⟩, M={M} Atoms', fontsize=14)
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
    plt.show()

# Parameters
M = 1500  # Number of atoms - try with a larger number
N = 10   # Initial field excitation number

# Run simulation
entropies_B = simulate_modified_jaynes_cummings(M, N)

# Plot results
plot_entropy(entropies_B, M, N)
