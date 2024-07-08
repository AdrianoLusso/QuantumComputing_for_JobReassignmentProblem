import random as r
import json
import numpy as np
import copy


class JRPRandomGenerator:
    '''  
    This class implements a random generator for Job Reassigment Problem (JRP) instances.
    
    Each instance is formed by:      
        - num_agents
        - num_vacnJobs
            number of vacant jobs
        - priorityWeightCoeff
            a number for giving a weight to the final priority of a possible reassigment to job 'j'
            for agent 'i':  x_ij
        - affinityWeightCoeff
            a number for giving a weight to the final priority for a x_ij
        - penalty1
            the penalty factor for the first penalty of the problem:
                each vacant job i can be done by at most one agent
        - penalty2
            the penalty factor for the second penalty of the problem:
                each agent j can be reasigned to at most one job
        - agents
            a list of the agents
        - assgJobs
            a list of the currently assigned jobs
        - vacnJobs
            a list of the vacant jobs
        - agents_assgJobs
            a list where, for an integer 'i' in index 'j', means that the agent 'j' is currently assigned to
            job 'i'
        - agents_assgJobsAfinnity
            a matrix where, for an number 'a' in index ('j','i'), means that the agent 'j' has an affinity 'a'
            for the assigned job 'i'
        - agents_vacnJobsAfinnity
            a matrix where, for a number 'a' in index ('j','i'), means that the agent 'j' has an affinity 'a'
            for the vacant job 'i'
        - assgJobs_agents
            a list where, for an integer 'j' in index 'i', means that the agent 'j' is currently assigned to
            job 'i'
        - assgJobs_priority
            a lsit where, for a integer 'p' in index 'i', means that assigned job 'i' has priority 'p'
        - vacnJobs_priority
            a lsit where, for a integer 'p' in index 'i', means that vacant job 'i' has priority 'p'
        
        - weighted_priority_diff_matrix
            a matrix where, for an integer 'p_diff' in index ('i','j'), represents the priority difference
            'p_diff' of ressigning the agent 'j' from his current work into the vacant job 'i'
        - weighted_affinity_diff_matrix
            a matrix where, for an integer 'a_diff' in index ('i','j'), represents the affinity difference
            'p_diff' of ressigning the agent 'j' from his current work into the vacant job 'i'
        
        - allBinaryVariables:
            a list of all the binary variables for a QUBO formulation  
    '''


    def __init__(self,num_agents=None,num_vacnJobs=None,priorityWeightCoeff=1,affinityWeightCoeff=1,
        penalty1=2,penalty2=2,max_assgJobs_priority = 3,control_restrictions=True):
        '''
          Parameters (described at the first comment of the class):
            
            - num_agents
            - num_vacnJobs
            - prorityWeightCoeff (optional)
            - affinityWeightCoeff (optinal)
            - penalty1 (optional)
            - penalty2 (optional)
            - max_assgJobs_priority
                an integer M describing the maximum priority that an assigned job could have, and the minimum
                priority a vacant job could have. That is, every assigned job can have priority in [1,M] and
                every vacant job can have priority in [M,5]
            - controlRestrictions
                if True, the generator controls restrictions over number of agents and vacancy jobs
          '''
          
        self.num_agents = num_agents
        self.num_vacnJobs = num_vacnJobs
        self.priorityWeightCoeff = priorityWeightCoeff
        self.affinityWeightCoeff = affinityWeightCoeff
        self.penalty1 = penalty1
        self.penalty2 = penalty2
        self.max_assgJobs_priority = max_assgJobs_priority
        self.control_restrictions = control_restrictions
        
        # priorities range and affinity maximum value
        self.priorities = [1,2,3,4,5]
        self.affinity_maximum = 0.5

    def generate_random_instance(self):
        '''
        This method generate a random JRP instance making use of the parameters given at __init__

        Return:
            - a json with the instance attributes
        '''
        # controling restrictions over number of agents and vacancy jobs
        self.__number_restrictions_control()

        # create agents, their assign jobs and the vacant jobs
        agents,assgJobs,vacnJobs = self.__create_agents_and_jobs()

        # do the association of each agent and asshJob
        agents_assgJobs, assgJobs_agents = self.__agents_assgJobs(assgJobs)

        #random assignment of the agent affinity with each assigned job and vancant job
        # return a two matrixes
        agents_assgJobsAfinnity, agents_vacnJobsAfinnity = self.__affinity_matrixes(agents)

        # randomly choose the jobs priorities
        assgJobs_priority,vacnJobs_priority = self.__priorities()

        # calculate the priority difference matrix and affinity difference matrix
        weighted_priority_diff_matrix,weighted_affinity_diff_matrix = self.__diff_matrixes(
            assgJobs_priority,
            vacnJobs_priority,
            agents_assgJobs,
            agents_assgJobsAfinnity,
            agents_vacnJobsAfinnity)

        # prepare the data structure and save it
        allBinaryVariables = list(range(self.num_vacnJobs * self.num_agents))
        data = {
            "num_agents": self.num_agents,
            "num_vacnJobs": self.num_vacnJobs,
            "priorityWeightCoeff": self.priorityWeightCoeff,
            "affinityWeightCoeff": self.affinityWeightCoeff,
            "penalty1": self.penalty1,
            "penalty2": self.penalty2,
            "agents": agents,
            "assgJobs": assgJobs,
            "vacnJobs": vacnJobs,
            "agents_assgJobs": agents_assgJobs,
            "agents_assgJobsAfinnity":agents_assgJobsAfinnity,
            "agents_vacnJobsAfinnity":agents_vacnJobsAfinnity,
            "assgJobs_agents": assgJobs_agents,
            "assgJobs_priority": assgJobs_priority,
            "vacnJobs_priority": vacnJobs_priority,
            "weighted_priority_diff_matrix":weighted_priority_diff_matrix,
            "weighted_affinity_diff_matrix":weighted_affinity_diff_matrix,
            "allBinaryVariables": allBinaryVariables
        }
        json_data = json.dumps(data, indent=4)
        return json_data
    

    # auxiliary functions
    def __number_restrictions_control(self):
        ''' 
        controling restrictions over number of agents and vacancy jobs 
        '''
        if(self.control_restrictions and self.num_agents not in range(3,21)):
            raise ValueError('Not respecting the 3<= num_agents <=20 restriction')
        if(self.control_restrictions and self.num_vacnJobs not in range(5,31)):
            raise ValueError('Not respecting the 5<= num_vacnJobs <=30 restriction')
        
    def __create_agents_and_jobs(self):
        ''' 
        create agents, their assign jobs and the vacant jobs 

        Return:
            - the lists of the agents, assigned jobs and vacant jobs
        '''
        agents = list(range(self.num_agents))
        assgJobs = list(range(self.num_agents))
        vacnJobs = list(range(self.num_vacnJobs))
        return agents,assgJobs,vacnJobs

    def __agents_assgJobs(self,assgJobs):
        ''' 
        do the association of each agent and asshJob 

        Return:
            - the list agents_assgJobs and assgJobs_agents
        '''
        # it assign the assigned jobs to each agent randomly
        agents_assgJobs = assgJobs.copy()
        r.shuffle(agents_assgJobs)

        # this is the inverse list of agents_assgJobs, created before
        assgJobs_agents = [None] * len(agents_assgJobs)
        for i, value in enumerate(agents_assgJobs):
            assgJobs_agents[value] = i  

        return agents_assgJobs,assgJobs_agents
        
    def __affinity_matrixes(self,agents):
        ''' 
        do the affinity matrixes either between (agent,assgJob) and (agent,vacnJob) 

        Return:
            - affinity matrixes between agents and assigned jobs and agents and vacant jobs
        '''
        
        agents_assgJobsAfinnity = []
        agents_vacnJobsAfinnity = []
        for agent in agents:
            agents_assgJobsAfinnity.append(
                [round(r.random() * self.affinity_maximum,2) for _ in range(self.num_agents)]
            )
            agents_vacnJobsAfinnity.append(
                [round(r.random() * self.affinity_maximum,2) for _ in range(self.num_vacnJobs)]
            )

        return agents_assgJobsAfinnity,agents_vacnJobsAfinnity

    def __priorities(self):
        ''' 
        generate the priorities for the assgJobs and the vacnJobs 

        Return:
            - the lists of assigned jobs priorities and vacant jobs priorites
        '''
        assgJobs_priorities_domain = self.priorities[:self.max_assgJobs_priority]
        vacnJobs_priorities_domain = self.priorities[self.max_assgJobs_priority-1:]

        assgJobs_priority = r.choices(assgJobs_priorities_domain, k=self.num_agents)
        vacnJobs_priority = r.choices(vacnJobs_priorities_domain,weights=vacnJobs_priorities_domain, k=self.num_vacnJobs)
        #range(1, len(self.priorities) + 1)
        return assgJobs_priority,vacnJobs_priority

    def __diff_matrixes(self,
            assgJobs_priority,
            vacnJobs_priority,
            agents_assgJobs,
            agents_assgJobsAfinnity,
            agents_vacnJobsAfinnity):
        ''' 
        calculate the weighted priority difference matrix and weighted affinity difference matrix 

        Return:
            -The weighted differences matrixes for priority and forn affinity
        '''
        
        #priority difference matrix
        weighted_priority_diff_matrix = []
        for vacnJob,priority in enumerate(vacnJobs_priority):
            weighted_priority_diff_matrix.append([])
            for assgJob in agents_assgJobs:
                weighted_priority_diff_matrix[vacnJob].append(self.priorityWeightCoeff *(priority - assgJobs_priority[assgJob]))

        #affinity difference matrix
        weighted_affinity_diff_matrix = copy.deepcopy(agents_vacnJobsAfinnity)
        for agent,agent_list in enumerate(agents_vacnJobsAfinnity):
            for vacnJob in range(len(agent_list)):
                weighted_affinity_diff_matrix[agent][vacnJob] =  round(self.affinityWeightCoeff *
                    (weighted_affinity_diff_matrix[agent][vacnJob] - agents_assgJobsAfinnity[agent][agents_assgJobs[agent]])
                    ,2)
        weighted_affinity_diff_matrix = [list(row) for row in zip(*weighted_affinity_diff_matrix)]


        return weighted_priority_diff_matrix,weighted_affinity_diff_matrix