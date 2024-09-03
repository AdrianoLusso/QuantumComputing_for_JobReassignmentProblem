from JRPClassic import JRPClassic
from QAOASolver import QAOASolver

from openqaoa.backends import create_device

import json

class EvaluationQAOASolver(QAOASolver):
    '''
    TODO
    '''

    def __init__(self):
        pass

    def sample_workflows(self,files,filename,instance):
        '''
        '''
        optimized_params = files[filename][str(instance)]['result']['optimized']['angles']

        keys = files.keys()
        p = files[filename]['circuit_configuration']['p']

        samples = {
            'reused_instance':{
                'filename': filename,
                'instance': instance
            }
        }
        for key in keys:
            circuit_configuration = files[key]["circuit_configuration"]
            aux_p = circuit_configuration['p']
            
            if p != aux_p:
                continue

            samples[key] = {}
            
            evaluation_backend_configuration = files[key]["evaluation_backend_configuration"]
            for ins in range(5):
                if filename in key and str(instance) in str(ins):
                    continue
                
                instance_dict = files[key][str(ins)]["instance"]
                #print('ssd')
                
                jrp = JRPClassic(instance_dict)
                opt_standard_solution = files[key][str(ins)]["opt_standard_solution"]
                opt_standard_gain =jrp.calculate_standard_gain(files[key][str(ins)]["opt_standard_solution"])
                
                jrp_ising = super().get_jrp_ising(jrp)

                device = create_device(location='local', name='qiskit.shot_simulator')
                qaoa = super().create_and_configure_qaoa(device,circuit_configuration,
                                          evaluation_backend_configuration,
                                          jrp_ising)

                qaoa.compile(jrp_ising)
                #qaoa.backend.backend_simulator = AerSimulator(precision='single')
                
                preliminary_qubo_solutions = qaoa.evaluate_circuit(optimized_params)['measurement_results']
                final_standard_solution,final_standard_gain = super().filter_qubo_solutions(jrp,preliminary_qubo_solutions)
                
                if opt_standard_gain == 0:
                    approximation_ratio = None
                else:
                    approximation_ratio = final_standard_gain / opt_standard_gain
                
                sample = {
                    'approximation ratio':approximation_ratio,
                    'opt_standard_solution':opt_standard_solution,
                    'final_standard_solution':final_standard_solution,
                }

                samples[key][str(ins)] = sample
                # save the json and compress it
                with open('./EvaluationQAOASolver_%s_instance%s.json'%(str(filename),str(instance)), 'w', encoding='utf-8') as file:
                    json.dump(samples, file, ensure_ascii=False, indent=4)
