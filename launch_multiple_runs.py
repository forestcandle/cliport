import os
from optparse import OptionParser
import itertools

if __name__ == '__main__':
    args=[]
    for sentance_ind in range(10):
        for id_inds in itertools.combinations(list(range(8)), 2):
            args.append(f"adjective_sentance_num={sentance_ind} identity_group_num_0={id_inds[0]} identity_group_num_1={id_inds[1]}")
    
    for ind in range(len(args)):
        os.system(f'sbatch run_cliport.sh "{args[ind]}"')
    os.system("squeue -u wagnew3")