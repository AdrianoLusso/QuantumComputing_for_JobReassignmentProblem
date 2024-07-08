from openqaoa import QAOA
from openqaoa import QUBO
from openqaoa.algorithms import QAOAResult
from openqaoa.backends import create_device

#import sys
#import os
#import asyncio
import json

from JRPClassic import JRPClassic
from JRPRandomGenerator import JRPRandomGenerator


class TestQAOASolver:
    '''
    TODO
    '''

    def __init__(self):
        '''
        TODO
        '''
        pass
    #n_shots_for_optimization,n_shots_for_validation
    def sample_workflows(self,configuration_name,n_samples,jrp_init_configuration,circuit_configuration,
                         optimizer_configuration,optimization_backend_configuration,
                         evaluation_backend_configuration,device=None):
        '''
        this method do a sample of 'testQAOAsolver' workflows and save them in json files.

        Parameters:
            - configuration_name
                a string with the name of the configuration, so as to identify the json file
            - n_samples            
            - jrp_init_configuration
                a dictionary with the parameters for the JRPRandomGenerator. This should specify:
                    ^ num_agents: integer
                    ^ num_vacnJobs: integer
                    ^ control_restrictions: boolean
            - circuit_configuration
                a dictionary with the parameters for the set_circuit_properties() method of a QAOA object. This should specify:
                    ^ p
                    ^ param_type
                    ^ init_type
                    ^ mixer_hamiltonian
                more info about possible parameters values in OpenQAOA docs.
            - optimized_configuration
                a dictionary with the parameters for the set_classical_optimizer() method of a QAOA object. This should specify:
                    ^ method
                    ^ maxfev
                    ^ tol
                    ^ optimization_progress: boolean
                    ^ cost_progress: boolean
                    ^ parameter_log: boolean
                more info about possible parameters values in OpenQAOA docs.
            - n_shots_for_optimization
                the integer number of shots for the quantum ansatz during the quantum-classical optimization loop
            - n_shots_for_validation
                the integer number of shots for the quantum ansatz during the validation of the optimized variational parameters.
                The number of this parameters will be the number of solutions that will be evaluated as possible solutions.
            - device
                the device where QAOA will be run.
        '''
        
        if device is None:
            device = create_device(location='local', name='qiskit.shot_simulator')

        # create the random JRP instances generator
        jrp_gen = JRPRandomGenerator(**jrp_init_configuration)

        # starts the sampling
        samples = {'configuration':jrp_init_configuration}
        for sample_index in range(n_samples):
            #create a random instance
            jrp_instance_dict = jrp_gen.generate_random_instance()
            jrp_instance_dict = json.loads(jrp_instance_dict)
            #print(jrp_instance_dict)
            jrp = JRPClassic(jrp_instance_dict)
            
            # get the important results
            print('running sample ',sample_index)
            approximation_ratio,standard_gain_difference, ising_cost_difference,opt_standard_solution, final_standard_solution,qaoa_result = self.__run_workflow(
                jrp,circuit_configuration,
                optimizer_configuration,
                optimization_backend_configuration,
                evaluation_backend_configuration,
                device
            )

            # creates the sample dats structure and saves it
            sample = {
                'instance': jrp_instance_dict,
                'approximation_ratio':approximation_ratio,
                'standard_gain_difference':standard_gain_difference,
                'ising_cost_difference':ising_cost_difference,
                'opt_standard_solution':opt_standard_solution,
                'final_standard_solution':final_standard_solution,
                'result':qaoa_result.asdict()
            }
            samples[sample_index] = sample
            with open('./conf%s.json'%(str(configuration_name)), 'w', encoding='utf-8') as file:
                json.dump(samples, file, ensure_ascii=False, indent=4)
    

    def __run_workflow(self,jrp,circuit_configuration, optimizer_configuration,
                       optimization_backend_configuration,
                       evaluation_backend_configuration,device):
        '''
        This method runs A 'testQAOAsolver' workflow for a particular JRP instance.

        Parameters:
            - jrp
                a JRP object
            - circuit_configuration
                a dictionary with the parameters for the set_circuit_properties() method of a QAOA object. This should specify:
                    ^ p
                    ^ param_type
                    ^ init_type
                    ^ mixer_hamiltonian
                more info about possible parameters values in OpenQAOA docs.
            - optimizer_configuration
                a dictionary with the parameters for the set_classical_optimizer() method of a QAOA object. This should specify:
                    ^ method
                    ^ maxfev
                    ^ tol
                    ^ optimization_progress: boolean
                    ^ cost_progress: boolean
                    ^ parameter_log: boolean
                more info about possible parameters values in OpenQAOA docs.
            - n_shots_for_optimization
                the integer number of shots for the quantum ansatz during the quantum-classical optimization loop
            - n_shots_for_validation
                the integer number of shots for the quantum ansatz during the validation of the optimized variational parameters.
                The number of this parameters will be the number of solutions that will be evaluated as possible solutions.
            - device
                the device where QAOA will be run.

        Return:
            approximation_ratio
                a 0 to 1 ratio showing how approximate to the optimal solution was the solution getted from the QAOA.
            standard_gain_difference
                absolute difference between the gain of the initial standard solution (no reassignment done) and the final standard
                solution (the reassignments found after QAOA optimization)
            ising_cost_difference
                the absolute difference between the initial ising expectation value and the final ising expectation value
            opt_standard_solution
                optimal solution, in standard formulation, found by brute force
            final_standard_solution
                final solution, in standard formulation, found by QAOA optimization
            qaoa_result
                a QAOA_Result object from OpenQAOA
            

        '''
        # initial configuration
        initial_standard_solution = [-1 for i in range(jrp.instance_dict['num_agents'])]
        initial_standard_gain = jrp.calculate_standard_gain(initial_standard_solution)

        #optimal configuration
        opt_standard_solution,opt_standard_gain = jrp.solve_standard_with_bruteforce(debug_every=100000)
        
        # final configuration - after QAOA optimization
        final_standard_solution,final_standard_gain,initial_ising_cost,final_ising_cost,qaoa_result = self.__run_qaoa_workflow(
            jrp,
            circuit_configuration,
        optimizer_configuration, 
        optimization_backend_configuration,
        evaluation_backend_configuration,device
        )

        if opt_standard_gain == 0:
            approximation_ratio = None
        else:
            approximation_ratio = final_standard_gain / opt_standard_gain
        standard_gain_difference = final_standard_gain - initial_standard_gain
        ising_cost_difference = final_ising_cost - initial_ising_cost
        
        # TODO convegence curve 

        return approximation_ratio,standard_gain_difference,ising_cost_difference,opt_standard_solution,final_standard_solution,qaoa_result

    def __run_qaoa_workflow(self,jrp,circuit_configuration, optimizer_configuration, optimization_backend_configuration,
                       evaluation_backend_configuration,device):
        '''
        this  methods runs the QAOA workflow (sub-workflow of the complete 'testQAOASolver') for a particular JRP instance

        Parameters:
            -jrp
                a JRP object
            - circuit_configuration
                a dictionary with the parameters for the set_circuit_properties() method of a QAOA object. This should specify:
                    ^ p
                    ^ param_type
                    ^ init_type
                    ^ mixer_hamiltonian
                more info about possible parameters values in OpenQAOA docs.
            - optimizer_configuration
                a dictionary with the parameters for the set_classical_optimizer() method of a QAOA object. This should specify:
                    ^ method
                    ^ maxfev
                    ^ tol
                    ^ optimization_progress: boolean
                    ^ cost_progress: boolean
                    ^ parameter_log: boolean
                more info about possible parameters values in OpenQAOA docs.
            - n_shots_for_optimization
                the integer number of shots for the quantum ansatz during the quantum-classical optimization loop
            - n_shots_for_validation
                the integer number of shots for the quantum ansatz during the validation of the optimized variational parameters.
                The number of this parameters will be the number of solutions that will be evaluated as possible solutions.
            - device
                the device where QAOA will be run.

        Return:
            - final_standard_solution
                final solution, in standard formulation, found by QAOA optimization
            - final_standard_gain
                the gain asociated to the final standard solution
            - initial_ising_cost
            - final_ising_cost
            - qaoa_result
                a QAOA_Result object from OpenQAOA
        '''
        # create the QUBO in openqaoa
        jrp.to_qubo()
        terms,weights = jrp.to_openqaoa_format()
        terms,weights = QUBO.convert_qubo_to_ising(len(jrp.instance_dict['allBinaryVariables']), terms, weights)
        jrp_ising = QUBO(n=len(jrp.instance_dict['allBinaryVariables']),terms=terms,weights=weights)
        
        # create QAOA solver and configurations
        qaoa = QAOA()
        #device = create_device(location='local', name='qiskit.shot_simulator')
        qaoa.set_device(device)
        qaoa.set_circuit_properties(**circuit_configuration)
        qaoa.set_backend_properties(**optimization_backend_configuration)
        qaoa.set_classical_optimizer(**optimizer_configuration)
        
        #compile
        qaoa.compile(jrp_ising)

        # get the initial ising cost, using the initial variational parameters
        initial_ising_cost = qaoa.evaluate_circuit(qaoa.variate_params.asdict())['cost']

        # do the QAOA optimization
        qaoa.optimize()
        qaoa_result = qaoa.result
        
        # get the final ising cost, using the optimized variational parameters
        final_ising_cost = qaoa_result.optimized['cost']

        # create a new QAOA for sampling bitstrings using the optimized variational parameters
        qaoa = QAOA()
        qaoa.set_device(device)
        qaoa.set_circuit_properties(**circuit_configuration)
        qaoa.set_backend_properties(**evaluation_backend_configuration)
        qaoa.compile(jrp_ising)
        preliminary_qubo_solutions = qaoa.evaluate_circuit(qaoa_result.optimized['angles'])['measurement_results']

        # only takes the feasible solutions
        final_qubo_solutions = []
        for solution in preliminary_qubo_solutions.keys():
            if jrp.is_qubo_feasible(solution):
                final_qubo_solutions.append(solution)
        #print(final_qubo_solutions)

        # translate the solutions from QUBO format to standard format and calculate its gain.
        # store the pair (standardSolution,gain) with the maximal gain
        final_standard_gain = float('-inf')
        final_standard_solution = [-2 for i in range(jrp.instance_dict['num_agents'])]
        for qubo_solution in final_qubo_solutions:
            standard_solution = jrp.quboSolution_to_standardSolution(qubo_solution)
            standard_gain = jrp.calculate_standard_gain(standard_solution)
            
            if final_standard_gain < standard_gain:
                final_standard_solution = standard_solution
                final_standard_gain = standard_gain
        
        return final_standard_solution,final_standard_gain,initial_ising_cost,final_ising_cost,qaoa_result