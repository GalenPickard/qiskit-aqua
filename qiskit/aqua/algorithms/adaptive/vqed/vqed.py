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

def state_overlap(self, num_qubits):
    """Returns a state overlap circuit as a cirq.Circuit."""
    # declare a circuit
    qr = QuantumRegister(2 * num_qubits)
    cr = ClassicalRegister(2 * num_qubits)
    circ = QuantumCircuit(qr, cr, name="state_overlap_circuit")

    # gates to perform
    def bell_basis_gates(i):
        bell_qr = QuantumRegister(2)
        bell_circ = QuantumCircuit(bell_qr,
                                   name="bell-basis")
        bell_circ.cx(bell_qr[0], bell_qr[1]),
        bell_circ.h(bell_qr[0])

        return bell_circ

    # add the bell basis gates to the circuit
    for i in range(num_qubits):
        circ.append(bell_basis_gates(i).to_instruction(), [
                    qr[i], qr[ii + num_qubits]])

    # measurements
    qubits_to_measure = qr[:num_qubits]
    cbits_to_measure = cr[:num_qubits]

    circ.barrier()
    circ.measure(qubits_to_measure, cbits_to_measure)
    return circ

def state_overlap_postprocessing(self, output):
        """Does the classical post-processing for the state overlap algorithm.

        Args:
            output [type: np.array]
                The output of the state overlap algorithm.

                The format of output should be as follows:
                    vals.size = (number of circuit experiments
                                 number of qubits being measured)

                    the ith column of vals is all the measurements on the
                    ith qubit. The length of this column is the number
                    of times the circuit has been run.

        Returns:
            Estimate of the state overlap as a float
        """
        # =====================================================================
        # constants and error checking
        # =====================================================================

        # number of qubits and number of experimental shot of the circuit
        (nshots, nqubits) = output.shape
        print(output)

        # check that the number of qubits is even
        assert nqubits % 2 == 0, "Input is not a valid shape."

        # initialize variable to hold the state overlap estimate
        overlap = 0.0

        # =====================================================================
        # postprocessing
        # =====================================================================

        # loop over all the bitstrings produced by running the circuit
        shift = nqubits // 2
        for z in output:
            parity = 1
            pairs = [z[i] and z[ii + shift] for ii in range(shift)]
            # DEBUG
            for pair in pairs:
                parity *= (-1)**pair
            overlap += parity

        return overlap / nshots

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
    qc = QuantumCircuit(qr, cr, name="swap_circuit")

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
    qc = QuantumCircuit(qr, cr, name="dip_circuit")

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
    qc = QuantumCircuit(qr, cr, name="pdip_circuit")

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
