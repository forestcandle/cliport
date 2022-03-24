import os
from optparse import OptionParser
import itertools

if __name__ == '__main__':
    args=[]
    num_sentances_per=1
    for sentance_ind in [28, 137, 0, 50, 51, 52, 77, 78, 79, 83, 84, 85, 56, 57, 58]:
        args.append(f"command_string_min={sentance_ind} command_string_max={sentance_ind+num_sentances_per} num_processes=20")
    
    for ind in range(len(args)):
        os.system(f'sbatch run_cliport.sh "{args[ind]}"')
    os.system("squeue -u wagnew3")