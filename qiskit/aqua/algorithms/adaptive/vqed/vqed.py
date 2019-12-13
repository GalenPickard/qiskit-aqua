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
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.initial_states import VarFormBased

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
            var_form (VariationalForm): parametrized variational form.
            optimizer (Optimizer): the classical optimization algorithm.
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
        self._total_qubits = 2 * num_qubits  # pure states
        self._qubits = QuantumRegister(self._total_qubits)
        self._cbits = QuantumClassical(self._total_qubits)
        self._ret = None
        self._eval_time = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback
        if initial_point is None:
            self._initial_point = var_form.preferred_init_points
        self._eval_count = 0
        self._var_form_params = ParameterVector(
            'θ', self._var_form.num_parameters)

        self._state_prep_circ = None
        self._unitary_circ = None
        self._dip_test_circ = None
        self._purity = None
        self._parameterized_circuits = None

    def get_num_qubits(self):
        """Returns the number of qubits in the circuit."""
        return self._num_qubits

    # =========================================================================
    # methods to clear/reset circuits
    # =========================================================================

    def clear_state_prep_circ(self):
        """Sets the state prep circuit to be a new, empty circuit."""
        self.state_prep_circ = QuantumCircuit(self._total_qubits)

    def clear_unitary_circ(self):
        """Sets the unitary circuit to be a new, empty circuit."""
        self.unitary_circ = QuantumCircuit(self._total_qubits)

    def clear_dip_test_circ(self):
        """Sets the dip test circuit to be a new, empty circuit."""
        self.dip_test_circ = QuantumCircuit(self._total_qubits)

    def state_prep(self, params, depth=1, copy=0):
        """Adds the parametrized state prep circuit to self.state_prep_circ.

        input:
            params [type: list<float>]
                self._num_qubits x (depth + 1) * 2  list of floats corresponding to the
                rotation angles for RyRz  linear entanglement variational form

            copy [type: int, 0 or 1, default value = 0]
                the copy of the state rho to perform the state prep
                circuit on.

        modifies:
            self._state_prep_circ
        """
        # error check on inputs
        assert len(params) == self._num_qubits * (depth + 1) * 2

        # =====================================================================
        # initial variational state
        # =====================================================================

        state_prep_var_form = RYRZ(
            num_qubits, depth=depth, entanglement='linear')

        # =====================================================================

        self._state_prep_circ = state_prep_var_form.construct_circuit(params)

        # =====================================================================
        # unitary ansatz
        # =====================================================================

    def unitary_ansatz(self, params, depth=1, copy=0):
        """Adds the parametrized unitary circuit to self.unitary_circ

          input:
              params [type: list<float>]
                  self._num_qubits x (depth + 1/2) * 2  list of floats corresponding to the
                  rotation angles for Ry linear entanglement variational form

              copy [type: int, 0 or 1, default value = 0]
                  the copy of the state rho to perform the state prep
                  circuit on.

          modifies:
              self._unitary_circ
          """

        assert len(params) == self._var_form.num_parameters

        unitary_var_form = RY(
            num_qubits, depth=depth, entanglement='linear')

        self._unitary_circ = unitary_var_form.construct_circuit(params)

    def dip_circuit(self):
        """
        Construct DIP Test over 2n qubits

        Args:
            num_qubits (int): number of qubits in each of the states to be compared
        Returns:
            QuantumCircuit: the circuit
        """

        num_qubits = self._num_qubits
        total_num_qubits = self._total_qubits
        qr = QuantumRegister(total_num_qubits)
        cr = ClassicalRegister(total_num_qubits)
        qc = QuantumCircuit(qr, cr, name="dip_circuit")

        for i in range(num_qubits):
            qc.cx(qr[i + num_qubits], qr[i])
        qc.barrier(qr)

        for i in range(num_qubits):
            qc.measure(qr[i], cr[i])
        return qc

    def swap_circuit(self):
        """
        Construct Destructive Swap Test over 2n qubits

        Args:
            num_qubits (int): number of qubits in each of the states to be compared
        Returns:
            QuantumCircuit: the circuit
        """
        total_num_qubits = self._total_qubits
        num_qubits = self._num_qubits
        qr = QuantumRegister(total_num_qubits)
        cr = ClassicalRegister(total_num_qubits)
        qc = QuantumCircuit(qr, cr)

        for i in range(num_qubits):
            qc.cx(qr[i + num_qubits], qr[i])
            qc.h(qr[i + num_qubits])
        qc.barrier(qr)
        qc.measure(qr, cr)
        return qc

    def pdip_circuit(self, pdip_qubit_indices):
        """
        Construct PDIP Test over 2n qubits
        Args:
            num_qubits (int): number of qubits in each of the states to be compared
            pdip_qubit_indices (iterable[int]): qubit indices for which DIP test should be used
        Returns:
            QuantumCircuit: the circuit
        """
        num_qubits = self._num_qubits
        total_num_qubits = self._total_num_qubits
        qr = QuantumRegister(total_num_qubits)
        cr = ClassicalRegister(total_num_qubits)
        qc = QuantumCircuit(qr, cr)

                for i in range(num_qubits):
            qc.cx(qr[i + num_qubits], qr[i])
        qc.barrier(qr)

        all_qubit_indices = set(range(num_qubits))
        qubit_indices_to_hadamard = list(
            all_qubit_indices - set(pdip_qubit_indices)
        )
        hadamard_indices = [i + num_qubits
                            for i in qubit_indices_to_hadamard]

        swap_indices = hadamard_indices + qubit_indices_to_hadamard
        dip_n = len(pdip_qubit_indices)
        swap_n = len(swap_indices)

        dip_qc = QuantumCircuit(len(dip_n), name="dip_circuit")
        dip_qc.measure(np.arange(dip_n), np.arange(dip_n))

        swap_qc = QuantumCircuit(swap_n, name="swap_circuit")
        swap_qc.h(np.arange(swap_n))

        if swap_n > 0:
            swap_qc.measure(np.arange(swap_n), np.arange(swap_n))

        qc.append(dip_qc.to_instruction(), pdip_qubit_indices)
        qc.append(swap_qc.to_instruction(), swap_indices)

        return qc

    def dip_test(self):
        dip_qc = self.dip_circuit()
        self._dip_test_circ = dip_qc

    def pdip_test(self):
        pdip_qc = self.pdip_circuit()
        self._dip_test_circ = pdip_qc



    def _run(self):
        self._eval_count = 0
        self._ret = self.find_minimum(initial_point=self.initial_point,
                                      var_form=self.var_form,
                                      cost_fn=self.qcost,
                                      optimizer=self.optimizer)
        if self._ret['num_optimizer_evals'] is not None and self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        self.cleanup_parameterized_circuits()
        return self._ret

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

    def get_optimal_vector(self):
        pass

    def compute_purity(self,
                       simulator=Aer.get_backend('qasm_simulator'),
                       nshots=10000):
        """Computes and returns the (approximate) purity of the state."""
        # get the circuit without the diagonalizing unitary
        num_qubits = self._num_qubits
        total_num_qubits = self._total_qubits
        qr = QuantumRegister(total_num_qubits)
        cr = ClassicalRegister(total_num_qubits)

        circuit = QuantumCircuit(qr, cr)

        circuit.append(self.state_prep_circ.to_instruction(),
                qr[:num_qubits])

        circuit.append(self.state_overlap().to_instruction(),
                qr[num_qubits: total_num_qubits])

        # DEBUG
        print("I'm computing the purity as per the circuit:")
        print(circuit)
        results = simulator.execute(circuit, nshots=nshots, memory=True)
        memory = results.get_memory('dip_circuit')
        counts = _convert_to_int_arr(memory)
        self.purity = self.state_overlap_postprocessing(counts)

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
            pairs = [z[ii] and z[ii + shift] for ii in range(shift)]
            # DEBUG
            for pair in pairs:
                parity *= (-1) ** pair
            overlap += parity

    def c1_resolved(self,
           params,
           simulator=Aer.get_backend('qasm_simulator'),
           nshots=1000):
        """Computes c_1 term of the cost function"""

        if not self.purity:
            self.compute_purity()

        # run the circuit
        result = self.run_resolved(params, simulator, nshots)
        memory = result.get_memory('dip_circuit')
        counts = _convert_to_int_arr(memory)

        # compute the overlap and return the objective function
        overlap = counts[0] / repetitions if 0 in counts.keys() else 0
        return purity - overlap


    def c2_resolved(self, params, simulator=Aer.get_backend('qasm_simulator'), nshots=1000):
        """Returns the objective function as computed by the PDIP Test."""
        # make sure the purity is computed
        if not self.purity:
            self.compute_purity()

        # store the overlap
        ov = 0.0

        for j in range(self._num_qubits):
            # do the appropriate pdip test circuit
            self.clear_dip_test_circ()
            self.pdip_circuit([j])

            # DEBUG
            print("j =", j)
            print("PDIP Test Circuit:")
            print(self.dip_test_circ)

            # run the circuit
            result = self.run_resolved(angles, simulator, nshots)

            # get the measurement counts
            dipmemory = result.get_memory('dip_circuit')
            pdipmemory = result.get_memory('pdip_circuit')

            dipcounts = self._convert_to_int_arr(dipmemory)
            pdipcount = self._convert_to_int_arr(pdipmemory)

            # postselect on the all zeros outcome for the dip test measurement
            mask = self._get_mask_for_all_zero_outcome(dipcounts)
            toprocess = pdipcount[mask]

            # do the state overlap (destructive swap test) postprocessing
            overlap = self.state_overlap_postprocessing(toprocess)

            # DEBUG
            print("Overlap = ", overlap)

            # divide by the probability of getting the all zero outcome
            prob = len(np.where(mask == True)) / len(mask)
            counts = result.get_counts('dip_circuit')
            prob = counts[0] / repetitions if 0 in counts.keys() else 0.0

            assert 0 <= prob <= 1
            print("prob =", prob)

            overlap *= prob
            print("Scaled overlap =", overlap)
            print()
            ov += overlap

        return self_purity - ov / self._num_qubits


    def state_overlap(self):
        """Returns a the state overlap circuit as a cirq.Circuit."""
        # declare a circuit
        num_qubits = self._num_qubits
        total_num_qubits = self._total_num_qubits
        circuit = QuantumCircuit(total_num_qubits)


        # gates to perform
        def bell_basis_gates():
            qc = QuantumCircuit(2) 
            qc.cx(0, 1)
            qc.h(0)

            return qc

        # add the bell basis gates to the circuit
        for i in range(num_qubits):
            circuit.append(
                bell_basis_gates().to_instruction(), [i, i + num_qubits]
            )

        # measurements
        circuit.measure_all()


        return circuit


    def algorithm(self):
        num_qubits = self._num_qubits
        total_num_qubits = self._total_num_qubits
        qr = QuantumRegister(total_num_qubits)
        cr = ClassicalRegister(total_num_qubits)
        qc = QuantumRegister(qr, cr, name="vqsd")

        qc.append(self._state_prep_circ.to_instruction(), qr)
        qc.append(self._unitary_circ.to_instruction(), qr)
        qc.append(self._dip_test_circ.to_instruction(), qr)

        return qc


    def algorithm_resolved(self, params):
        qc = self.algorithm()

        if params is None:
            params = 2 * np.random.rand(12 *
                    self._var_form_params.num_parameters)
        binded_params = {self._var_form_params: params}
        qc.bind_paramaeters(binded_params)

        return qc

    def run(self,
            simulator=Aer.get_backend('qasm_simulator'),
            nshots=1000):
        """Runs the algorithm and returns the result.

        rtype: cirq.TrialResult
        """
        return simulator.execute(self.algorithm(), repetitions=repetitions)


    def qcost(params):
        # PDIP cost
        self.clear_dip_test_circ()
        pdip = self.c2_resolved(params, nshots=nshots)

        # DIP cost
        self..clear_dip_test_circ()
        self.dip_test()
        dip = self.c1_resolved(params, shots=nshots)

        # weighted sum
        obj = q * dip + (1 - q) * pdip

        return obj

    def run_resolved(self, params, simulator=Aer.get_backend('qasm_simulator'), nshots=1000):
        assert(params == self.var_form.num_paramaters)
        curr_param = {self._var_form_params: params}
        qc = self.algorithm_resolved(params)
        return simulator.execute(qc, nshots=nshots, memory=True)


    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError(
                "Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']

    @staticmethod
    def _convert_to_int_arr(memory):
        a = map(lambda x: list(x), memory)
        return np.array(list(a), dtype='int')
