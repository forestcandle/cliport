import os
from optparse import OptionParser
import itertools

if __name__ == '__main__':
    args=[]
    for sentance_ind in range([52, 137]):
        for id_ind in range([0,6]):
            args.append(f"adjective_sentance_num={sentance_ind} identity_group_num_0={id_ind}")
    
    for ind in range(len(args)):
        os.system(f'sbatch run_cliport.sh "{args[ind]}"')
    os.system("squeue -u wagnew3")