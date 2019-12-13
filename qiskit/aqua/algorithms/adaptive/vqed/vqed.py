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
from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm

from qiskit.aqua.components.optimizers import POWELL
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.components.initial_states import custom

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class VQED(VQAlgorithm):
    """
    The Variational Quantum Eigendecomposition algorithm.

    See https://arxiv.org/pdf/1810.10506.pdf
    """

    def __init__(self, optimizer, var_form, num_qubits,
                 initial_point=None, max_evals_grouped=1, callback=None):
        """Constructor.
        Args:
            optimizer (Optimizer): the classical optimization algorithm.
            var_form (VariationalForm): parametrized variational form.
            initial_point (numpy.ndarray): optimizer initial point.
            max_evals_grouped (int): max number of evaluations performed simultaneously
            callback (Callable): a callback that can access the intermediate data
                                 during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
        """
        var_form = var_form or RY(num_qubits)
        optimizer = optimizer or POWELL()
        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=None,
                         initial_point=initial_point)
        self._use_simulator_snapshot_mode = None
        self._num_qubits = num_qubits
        self._ret = None
        self._eval_time = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback
        if initial_point is None:
            self._initial_point = var_form.preferred_init_points
        self._eval_count = 0
        self._var_form_params = ParameterVector(
            'θ', self._var_form.num_parameters)

        self.state_prep_circ = None
        self.unitary_circ = None
        self.dip_test_circ = None
        self.purity = None
        self._parameterized_circuits = None

    def _run(self):
        self._eval_count = 0
        self._ret = self.find_minimum(initial_point=self.initial_point,
                                      var_form=self.var_form,
                                      cost_fn=self.cost_fn,
                                      optimizer=self.optimizer)
        if self._ret['num_optimizer_evals'] is not None and self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        self.cleanup_parameterized_circuits()
        return self._ret

    def _compute_purity(self,
                        simulator=Aer.get_backend('qasm_simulator'),
                        nshots=10000):
        """Computes the (approximate) purity of the state."""
        # get the circuit without the diagonalizing unitary
        num_qubits = self._num_qubits
        qr = QuantumRegister(2 * num_qubits)
        cr = ClassicalRegister(2 * num_qubits)

        circuit = QuantumCircuit(qr, cr)

        circuit.append(self.state_prep_circ.to_instruction(),
                       np.arange(2 * num_qubits))

        circuit.append(self._state_overlap_circuit(num_qubits).to_instruction(),
                       np.arange(2 * num_qubits))

        # DEBUG
        print("I'm computing the purity as per the circuit:")
        print(circuit)
        results = simulator.execute(circuit, nshots=nshots, memory=True)
        memory = results.get_memory('dip_circuit')
        counts = self._convert_to_int_arr(memory)
        self.purity = self.state_overlap_postprocessing(counts)

    def _init_state_circ(self, state_vector=None, circuit=None):
        assert state_vector is not None or circuit is not None, "Must specify either state_vector or circuit"

        if state_vector is not None:
            state_prep = custom.Custom(
                self.num_qubits, state_vector=state_vector)
        else:
            state_prep = custom.Custom(self.num_qubits, circuit=circuit)
        self.state_prep_circ = state_prep.construct_circuit()
        return self.state_prep_circ

    def c1(self,
           params,
           simulator=Aer.get_backend('qasm_simulator'),
           nshots=1000):
        """Computes c_1 term of the cost function"""

        if not self.purity:
            self._compute_purity()

        # run the circuit
        result = self.run(simulator, nshots)
        memory = result.get_memory('dip_circuit')
        counts = self._convert_to_int_arr(memory)

        # compute the overlap and return the objective function
        overlap = counts[0] / nshots if 0 in counts else 0
        return self.purity - overlap

    @staticmethod
    def state_overlap_postprocessing(output):
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
            pairs = [z[ii] and z[ii + shift] for ii in range(shift)]
            # DEBUG
            for pair in pairs:
                parity *= (-1) ** pair
            overlap += parity
        return overlap

    @staticmethod
    def _state_overlap_circuit(num_qubits):
        """
        Construct state overlap circuit over 2n qubits

        Args:
            num_qubits (int): number of qubits in each of the states to be compared
        Returns:
            QuantumCircuit: the circuit
        """
        # declare a circuit
        qr = QuantumRegister(2 * num_qubits)
        cr = ClassicalRegister(2 * num_qubits)
        circ = QuantumCircuit(qr, cr, name="state_overlap_circuit")

        # gates to perform
        def bell_basis_gates():
            bell_qr = QuantumRegister(2)
            bell_circ = QuantumCircuit(bell_qr,
                                       name="bell-basis")
            bell_circ.cx(bell_qr[0], bell_qr[1]),
            bell_circ.h(bell_qr[0])

            return bell_circ

        # add the bell basis gates to the circuit
        for i in range(num_qubits):
            circ.append(bell_basis_gates().to_instruction(), [
                qr[i], qr[i + num_qubits]])

        # measurements
        qubits_to_measure = qr[:num_qubits]
        cbits_to_measure = cr[:num_qubits]

        circ.barrier()
        circ.measure(qubits_to_measure, cbits_to_measure)
        return circ

    @staticmethod
    def swap_circuit(num_qubits):
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
            qc.cx(qr[i + num_qubits], qr[i])
            qc.h(qr[i + num_qubits])
        qc.barrier(qr)
        qc.measure(qr, cr)
        return qc

    @staticmethod
    def dip_circuit(num_qubits):
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
            qc.cx(qr[i + num_qubits], qr[i])
        qc.barrier(qr)
        for i in range(num_qubits):
            qc.measure(qr[i], cr[i])
        return qc

    @staticmethod
    def pdip_circuit(num_qubits, j):
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
            qc.cx(qr[i + num_qubits], qr[i])
            if i not in j:
                qc.h(qr[i + num_qubits])
        qc.barrier(qr)
        for i in range(num_qubits):
            qc.measure(qr[i], cr[i])
            if i not in j:
                qc.measure(qr[i + num_qubits], cr[i + num_qubits])
        return qc

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError(
                "Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']

    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")
        return self._var_form.construct_circuit(self._ret['opt_params'])

    # TODO: Implement optimal vector extraction when using a supported backend
    def get_optimal_vector(self):
        """Not yet implemented. Required due to VQAlgorithm abstract class."""
        pass

    @staticmethod
    def _convert_to_int_arr(memory):
        a = map(lambda x: list(x), memory)
        return np.array(list(a), dtype='int')
