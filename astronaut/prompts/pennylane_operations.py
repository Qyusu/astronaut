from textwrap import dedent


def get_pennylane_operations(lib_version: str = "0.39.0") -> str:
    if lib_version == "0.39.0":
        operations = dedent(
            """
            ## qml.BasisState
            - **Description**: Prepares a single computational basis state.
            - **Parameters**:
                - `state` (array[bool]): Binary array representing the basis state to prepare.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.CNOT
            - **Description**: The controlled-NOT operator.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.ControlledPhaseShift
            - **Description**: The controlled phase shift operation.
            - **Parameters**:
                - `phi` (float): Phase shift angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.ControlledQubitUnitary
            - **Description**: Applies an arbitrary fixed unitary operation, controlled by one or more wires.
            - **Parameters**:
                - `U` (array[complex]): The unitary matrix.
                - `control_wires` (Sequence[int]): The control wires.
                - `wires` (Sequence[int]): The target wires.

            ## qml.CRot
            - **Description**: Controlled arbitrary rotation operation.
            - **Parameters**:
                - `phi` (float): Rotation angle around the Z axis.
                - `theta` (float): Rotation angle around the Y axis.
                - `omega` (float): Rotation angle around the Z axis.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.CRX
            - **Description**: Controlled X rotation.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.CRY
            - **Description**: Controlled Y rotation.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.CRZ
            - **Description**: Controlled Z rotation.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.CSWAP
            - **Description**: Controlled SWAP operation.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.CY
            - **Description**: Controlled Y gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.CZ
            - **Description**: Controlled Z gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.DiagonalQubitUnitary
            - **Description**: Applies an arbitrary diagonal unitary matrix of dimension \(2^n\).
            - **Parameters**:
                - `D` (array[complex]): The diagonal unitary matrix.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.DoubleExcitation
            - **Description**: Double excitation rotation.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.DoubleExcitationMinus
            - **Description**: Double excitation rotation with a negative phase shift outside the subspace.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.DoubleExcitationPlus
            - **Description**: Double excitation rotation with a positive phase shift outside the subspace.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.ECR
            - **Description**: Echoed cross-resonance gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.Hadamard
            - **Description**: The Hadamard gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.Identity
            - **Description**: The identity operation.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.IsingXX
            - **Description**: Ising XX coupling gate.
            - **Parameters**:
                - `phi` (float): Coupling angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.IsingXY
            - **Description**: Ising (XX + YY) coupling gate.
            - **Parameters**:
                - `phi` (float): Coupling angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.IsingYY
            - **Description**: Ising YY coupling gate.
            - **Parameters**:
                - `phi` (float): Coupling angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.IsingZZ
            - **Description**: Ising ZZ coupling gate.
            - **Parameters**:
                - `phi` (float): Coupling angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.ISWAP
            - **Description**: The i-SWAP gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.MultiControlledX
            - **Description**: Multi-controlled Pauli X gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The control and target wires.

            ## qml.MultiRZ
            - **Description**: Arbitrary multi Z rotation.
            - **Parameters**:
                - `theta` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.OrbitalRotation
            - **Description**: Spin-adapted spatial orbital rotation.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.PauliX
            - **Description**: Pauli X operation.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.PauliY
            - **Description**: Pauli Y operation.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.PauliZ
            - **Description**: Pauli Z operation.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.PhaseShift
            - **Description**: Arbitrary single-qubit local phase shift.
            - **Parameters**:
                - `phi` (float): Phase shift angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.PSWAP
            - **Description**: Phase SWAP gate.
            - **Parameters**:
                - `phi` (float): Phase angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.QFT
            - **Description**: Applies the Quantum Fourier Transform (QFT).
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.QubitCarry
            - **Description**: Applies a QubitCarry operation to 4 input wires.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.QubitStateVector
            - **Description**: Initializes a quantum state using a state vector.  
            - **Note**: Deprecated and will be removed in version 0.40.
            - **Parameters**:
                - `state` (array[complex]): State vector.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.QubitSum
            - **Description**: Applies a QubitSum operation to 3 input wires.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.QubitUnitary
            - **Description**: Applies an arbitrary unitary matrix of dimension \(2^n\).
            - **Parameters**:
                - `U` (array[complex]): The unitary matrix.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.Rot
            - **Description**: Arbitrary single-qubit rotation.
            - **Parameters**:
                - `phi` (float): Rotation angle around the Z axis.
                - `theta` (float): Rotation angle around the Y axis.
                - `omega` (float): Rotation angle around the Z axis.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.RX
            - **Description**: Single-qubit rotation about the X axis.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.RY
            - **Description**: Single-qubit rotation about the Y axis.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.RZ
            - **Description**: Single-qubit rotation about the Z axis.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.S
            - **Description**: The single-qubit phase gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.SWAP
            - **Description**: The SWAP gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.T
            - **Description**: The T gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.Toffoli
            - **Description**: The Toffoli (CCNOT) gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.U1
            - **Description**: Arbitrary single-qubit U1 gate.
            - **Parameters**:
                - `phi` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.U2
            - **Description**: Arbitrary single-qubit U2 gate.
            - **Parameters**:
                - `phi` (float): Phase angle.
                - `lambda` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.U3
            - **Description**: Arbitrary single-qubit U3 gate.
            - **Parameters**:
                - `theta` (float): Rotation angle.
                - `phi` (float): Phase angle.
                - `lambda` (float): Rotation angle.
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.X
            - **Description**: The Pauli X gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.Y
            - **Description**: The Pauli Y gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.

            ## qml.Z
            - **Description**: The Pauli Z gate.
            - **Parameters**:
                - `wires` (Sequence[int]): The wires the operation acts on.
        """
        )
    else:
        raise ValueError(f"Unsupported PennyLane version: {lib_version}")

    return operations
