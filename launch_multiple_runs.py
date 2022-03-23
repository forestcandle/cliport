import os
from optparse import OptionParser
import itertools

if __name__ == '__main__':
    args=[]
    num_sentances_per=5
    for sentance_ind in range(0, 10, num_sentances_per):
        args.append(f"command_string_min={sentance_ind} command_string_max={sentance_ind+num_sentances_per}")
    
    for ind in range(len(args)):
        os.system(f'sbatch run_cliport.sh "{args[ind]}"')
    os.system("squeue -u wagnew3")