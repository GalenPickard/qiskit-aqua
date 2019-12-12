# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Variational Quantum Eigendecomposition algorithm """

import logging
import math
import numpy as np

from sklearn.utils import shuffle
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit import Aer

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class, AquaError
from qiskit.aqua.utils import get_feature_dimension
from qiskit.aqua.utils import map_label_to_class_name
from qiskit.aqua.utils import split_dataset_to_data_and_labels
from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


def _convert_to_int_arr(memory):
    a = map(lambda x: list(x), memory)
    return np.array(list(a), dtype='int')

 def c1(self,
                simulator=Aer.get_backend('qasm_simulator'),
                nshots=1000):
        """Computes c_1 term of the cost function"""

        if not self.purity:
            self.compute_purity()

        # run the circuit
        result = self.run(simulator, nshots)
        memory = result.get_memory('dip_circuit')
        counts = _convert_to_int_arr(memory)
        
        # compute the overlap and return the objective function
        overlap = counts[0] / repetitions if 0 in counts.keys() else 0
        return purity  - overlap


def swap_circuit(self, num_qubits):
    """
    Construct Destructive Swap Test over 2n qubits

    Args:
        num_qubits (int): number of qubits in each of the states to be compared
    Returns:
        QuantumCircuit: the circuit
    """
    qr = QuantumRegister(2 * num_qubits)
    cr = ClassicalRegister(2 * num_qubits)
    qc = QuantumCircuit(qr, cr)

    for i in range(num_qubits):
        qc.cx(qr[i], qr[i + num_qubits])
        qc.h(qr[i + num_qubits])

    qc.barrier(qr)
    qc.measure(qr, cr)
    return qc


def dip_circuit(self, num_qubits):
    """
    Construct DIP Test over 2n qubits

    Args:
        num_qubits (int): number of qubits in each of the states to be compared
    Returns:
        QuantumCircuit: the circuit
    """
    qr = QuantumRegister(2 * num_qubits)
    cr = ClassicalRegister(2 * num_qubits)
    qc = QuantumCircuit(qr, cr)

    for i in range(num_qubits):
        qc.cx(qr[i], qr[i + num_qubits])

    qc.barrier(qr)

    for i in range(num_qubits):
        qc.measure(qr[i], cr[i])
    return qc


def pdip_circuit(self, num_qubits, j):
    """
    Construct PDIP Test over 2n qubits

    Args:
        num_qubits (int): number of qubits in each of the states to be compared
        j (iterable[int]): qubit indices for which DIP test should be used
    Returns:
        QuantumCircuit: the circuit
    """
    qr = QuantumRegister(2 * num_qubits)
    cr = ClassicalRegister(2 * num_qubits)
    qc = QuantumCircuit(qr, cr)

    for i in range(num_qubits):
        qc.cx(qr[i], qr[i + num_qubits])
        if i not in j:
            qc.h(qr[i + num_qubits])

    qc.barrier(qr)

    for i in range(num_qubits):
        qc.measure(qr[i], cr[i])
        if i not in j:
            qc.measure(qr[i+num_qubits], cr[i+num_qubits])
    return qc
