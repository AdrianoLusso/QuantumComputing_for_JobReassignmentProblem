from openqaoa.qaoa_components import Hamiltonian,PauliOp
from openqaoa.algorithms import QAOAResult
import matplotlib.pyplot as plt
from JRPClassic import JRPClassic
import numpy as np
import json

def hamiltonian_from_dict(hamiltonian_dict):
    '''
    transforms a dictionary with the data to create a Hamiltonian object from OpenQAOA, into the respective object.

    Parameters:
        - hamiltonian_dict
            a dictionary with the hamiltonian information. The typical structure expected is the same as the 'cost_hamiltonian'
            attribute of a 'QAOAResult' object, from OpenQAOA.
    '''
    pauli_terms = []
    coeffs = []

    # creates the Pauli operations and its respective coefficients
    for term_dict,coeff in zip( hamiltonian_dict['terms'], hamiltonian_dict['coeffs']):
        pauli_str = term_dict['pauli_str']
        qubit_indices = term_dict['qubit_indices']
        pauli_op = PauliOp(pauli_str,qubit_indices)

        pauli_terms.append(pauli_op)
        coeffs.append(coeff)
    
    # creates de hamiltonian
    hamiltonian = Hamiltonian(pauli_terms,coeffs, hamiltonian_dict['constant'])
    return hamiltonian

def tensor3_for_approximation_ratio(file,q_result=None,create_cost_hamiltonian_object=True,
                                         size=5000,n_measures=100,directory=None,plot=False):
    '''
    creates a 3-tensor where, a point (X,Y,Z) is interpretated as follows:
    "After doing Z measures of the ansatz with the optimized parameters,
    there is a probability of Y that one of that measured solutions has
    an approximation ratio of X or bigger"

    Parameters:
        - file
        a dictionary representing a json file from a TestQAOASolver experiment
        - q_result
        - create_cost_hamiltonian_object
        - size
        - n_measures
        - directory
    '''
    
    # STEP 1: PARAMETERS CONTROLS
    # if the QAOAResult object is not given, it is taken from the json file
    if q_result is None:
        q_result = QAOAResult.from_dict(file['result'])
    # if it is indicated to create the Hamiltonian object for the cost hamiltonian
    if create_cost_hamiltonian_object:
        q_result.cost_hamiltonian = hamiltonian_from_dict(q_result.cost_hamiltonian)

    # STEP 2: INITIALIZE IMPORTANT VARIABLES
    #   - 'size' lowest cost bitstrings, in Ising formulation
    solutions_bitstring = q_result.lowest_cost_bitstrings(size)['solutions_bitstrings']
    #   - probabilities of measuring each of the bitstring
    probabilities = q_result.lowest_cost_bitstrings(size)['probabilities']
    #   - the JRP instance
    instance =  JRPClassic(file['instance'])
    #   - the gain of the optimal standard solution
    optimal_standard_solution_gain = instance.calculate_standard_gain(file['opt_standard_solution'])


    # STEP 3: CONVERTS ISING SOLUTIONS TO STANDARD ONES
    standard_solutions = []
    aux_probabilities = []
    for s,p in zip(solutions_bitstring,probabilities):
        try:
            # if the method raise an error, means that the solution is unfeasible, so it won't be appended to final list
            ss=instance.quboSolution_to_standardSolution(s,check_feasibility=True)
            standard_solutions.append(ss)
            aux_probabilities.append(p)
        except:
            pass
    probabilities = aux_probabilities

    # STEP 4: CALCULATE THE APPROXIMATION RATIOS FOR EACH LOWEST STANDARD SOLUTION
    approximation_ratios = []
    for s in standard_solutions:
        approx_ratio = instance.calculate_standard_gain(s)/optimal_standard_solution_gain
        approximation_ratios.append(round(approx_ratio,4))

    # STEP 5: SUM UP PROBABILITIES OF REPEATED APPROXIMATION RATIOS
    aux_dict = {}
    for ar,p in zip(approximation_ratios,probabilities):
        # if an aaproximation ratio is alredy in the dictionary, it won't replace the key value, instead, it will sum up the new probability
        if str(ar) in aux_dict.keys():
            aux_dict[str(ar)] = aux_dict[str(ar)] + p
        # else, it will initializate with the new probability
        else:
            aux_dict[str(ar)] = p
    approximation_ratios = [float(i) for i in list(aux_dict.keys())]
    probabilities = list(aux_dict.values())

    # STEP 6: calculate the Complementary Cumulative Distribution Function over the probabilities
    # https://en.wikipedia.org/wiki/Cumulative_distribution_function#Complementary_cumulative_distribution_function_(tail_distribution)
    ccdf = []
    for index in range(len(probabilities)):
        ccdf.append(round(np.sum(np.array(probabilities[:index+1])),4))

    # STEP 7: expand the ccdf for over n measures
    measures = range(n_measures)
    array_3D = np.zeros((n_measures, len(ccdf)))
    for i in range(array_3D.shape[0]):
        for j in range(array_3D.shape[1]):
            array_3D[i,j] = 1-(1-ccdf[j])** measures[i]

    # STEP 8: final plot
    X = approximation_ratios  
    Z = measures  
    Y = array_3D

    axis = {
        'X':X,
        'Z':list(Z),
        'Y':Y.tolist()
    }
    with open(directory, 'w') as f:
        json.dump(axis, f)

    '''
    plt.figure(figsize=(7, 4))
    plt.contourf(X,Z, Y, levels=20, cmap='viridis')
    plt.colorbar(label='Probability')
    plt.xlabel('minimum approximation ratio expected')
    plt.ylabel('Number of Measurements')
    #plt.title('Probability of Measuring a Solution with Approximation Error X or Less')
    plt.grid(True)
    plt.savefig(plot_directory)
    plt.show()
    plt.close()
    '''
    if plot:
        make_contour_plot(X,Y,Z)

def make_contour_plot(X,Z,Y,x_label=None,y_label=None,z_label=None,directory=None,ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    c = ax.contourf(X,Z, Y, levels=20, cmap='viridis')
    cbar = plt.colorbar(c,ax=ax,label='Probability')
    ax.set_xlabel('minimum approximation ratio expected')
    ax.set_ylabel('Number of Measurements')

    y_ticks = np.arange(0, 1100, 100)
    ax.set_yticks(y_ticks)
    x_ticks = np.arange(0.0, 1.1, 0.1)
    ax.set_xticks(x_ticks)

    #plt.title('Probability of Measuring a Solution with Approximation Error X or Less')
    ax.grid(True)
    if directory is not None:
        fig.savefig(directory)
    #plt.show()
    #plt.close()