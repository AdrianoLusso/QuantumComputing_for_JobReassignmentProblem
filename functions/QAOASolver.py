from abc import ABC, abstractmethod
from openqaoa import QUBO,QAOA


class QAOASolver:
    '''
    TODO
    '''

    def __init__(self):
        pass

    def get_jrp_ising(self,jrp):
        '''
        TODO
        '''
        jrp.to_qubo()
        terms,weights = jrp.to_openqaoa_format()
        terms,weights = QUBO.convert_qubo_to_ising(len(jrp.instance_dict['allBinaryVariables']), terms, weights)
        jrp_ising = QUBO(n=len(jrp.instance_dict['allBinaryVariables']),terms=terms,weights=weights)
        
        return jrp_ising
    
    def create_and_configure_qaoa(self,device,circuit_configuration,backend_configuration,
                                  ising,optimizer_configuration = None):
        '''
        TODO
        '''
        qaoa = QAOA()
        qaoa.set_device(device)
        qaoa.set_circuit_properties(**circuit_configuration)
        qaoa.set_backend_properties(**backend_configuration)
        if optimizer_configuration is not None:
            qaoa.set_classical_optimizer(**optimizer_configuration)
        qaoa.compile(ising)

        return qaoa
    
    def filter_qubo_solutions(self,jrp,qubo_solutions):
        '''
        TODO
        '''
        # only takes the feasible solutions
        final_qubo_solutions = []
        for solution in qubo_solutions.keys():
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

        return final_standard_solution,final_standard_gain