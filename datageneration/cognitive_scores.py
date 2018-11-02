
import numpy as np
import sys
module_utils_directory = "/Users/raphael.couronne/Programming/ARAMIS/Software/leasp/Leaspy_utils/"
sys.path.append(module_utils_directory)
from data_generator import Data_Generator, sigmoid



class CognitiveScores:
    def __init__(self, dim_time, dim_input):

        model = {}
        model['noise_std']=1e-2
        p_array = [0.4, 0.3, 0.5, 0.5,0.4,0.3,0.8,0.6]
        model["p"]=[p_array[i] for i in range(dim_input)]
        model["t"]=78
        v_array = [0.03, 0.08, 0.15, 0.2, 0.2, 0.1, 0.1,0.25]
        model["v"]= [v_array[i] for i in range(dim_input)]
        model['ksi_mean']=0
        model['ksi_std']=1
        model['tau_mean']=0
        model['tau_std']=3
        model['geodesic']=sigmoid

        # A matrix (1 source)
        a_v_array = [1, -1, 1, -1, 1, -1, 1, -1]
        a_v = [a_v_array[i] for i in range(dim_input)]
        model["a_matrix"]=(np.array(a_v)-np.array(a_v).dot(model["v"])/(np.array(model["v"]).dot(model["v"]))*np.array(model["v"]))
        model['sources_std']=2e-2

        # Cohort
        cohort = {}


        cohort['n_visits'] = dim_time
        cohort['duration'] = 6
        cohort['patient_sd_begin_age'] = 2
        cohort['patient_mean_begin_age']=80 - cohort['duration']/2.


        cohort['name']="PPMI-Clone-2modalities"
        cohort['n_patients']=int(1e6)

        self.data_generator = Data_Generator(cohort, model)
        self.patient = 0

    def sample(self):
        sample = self.data_generator.generate(self.patient)
        self.patient+=1
        return sample