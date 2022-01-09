import pickle
import numpy as np
import os
from optparse import OptionParser
import scipy.stats as st


def get_stats_for_run(runs_file):
    ''' Print out averages per identity per command. 
    
    args:
        runs_file: string, either run log or folder containing run logs.
        
    '''
    
    if os.path.isdir(runs_file):
        for file in os.listdir(runs_file):
            runs=pickle.load(open(runs_file, 'rb'))
            all_runs.append(runs)           
    else:
        all_runs=pickle.load(open(runs_file, 'rb'))
    
    # Parse data list into dict
    #cmd str, run num, run step
    data_dict={}
    for run in all_runs:
        cmd_str=run[2]
        if cmd_str not in data_dict:
            data_dict[cmd_str]={}
        run_num=run[0]
        if run_num not in data_dict[cmd_str]:
            data_dict[cmd_str][run_num]={}
        step_num=run[3]
        data_dict[cmd_str][run_num][step_num]=run
    
    # Compute placed rankings and frequencies for each cmd and identity
    rankings_dict={}
    placed_dict={}
    for cmd in data_dict:
        if cmd not in rankings_dict:
            rankings_dict[cmd]={}
            placed_dict[cmd]={}
        for run_num in data_dict[cmd]:
            ordered_runs=[data_dict[cmd][run_num][i] for i in range(len(data_dict[cmd][run_num]))]
        ids=[]
        raw_order=[]
        for obj_ind in range(4, len(ordered_runs[0]), 3):
            identity=ordered_runs[0][obj_ind][0]
            placed_status=np.array([ordered_runs[i][obj_ind+2] for i in range(len(ordered_runs))])
            placed=np.sum(placed_status)>0
            if identity[0] not in placed_dict[cmd]:
                placed_dict[cmd][identity[0]]={}
                rankings_dict[cmd][identity[0]]={}
            if identity[1] not in placed_dict[cmd][identity[0]]:
                placed_dict[cmd][identity[0]][identity[1]]=[]
                rankings_dict[cmd][identity[0]][identity[1]]=[]
            placed_dict[cmd][identity[0]][identity[1]].append(placed)
            ids.append(identity)
            if placed==1:
                raw_order.append(np.argwhere(placed_status)[0,0])
            else:
                raw_order.append(placed_status.shape[0])
        
        ordering=np.argsort(np.array(raw_order))
        for ind in range(ordering.shape[0]):
            if raw_order[ind]==placed_status.shape[0]:
                order=placed_status.shape[0]
            else:
                order=ordering[ind]
            identity=ids[ind]
            rankings_dict[cmd][identity[0]][identity[1]].append(order)
        u=0
    
    dicts=(rankings_dict, placed_dict)
    metric_names=("order object placed", "object placed")
    # Compute means and 90% CIs
    for d_ind in range(len(dicts)):
        data_dict=dicts[d_ind]
        for cmd in data_dict:
            for id_1 in data_dict[cmd]:
                for id_2 in data_dict[cmd][id_1]:
                    data=np.array(data_dict[cmd][id_1][id_2])
                    mean=np.mean(data)
                    low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                    high_err=mean+(mean-low_err)
                    print(f"{cmd} | {metric_names[d_ind]} | {id_1} | {id_2} | mean: {mean} CI: ({low_err}, {high_err})")

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--runs_file", dest="runs_file", default="/home/willie/github/cliport/cliport_quickstart/packing-unseen-google-objects-race-seq-cliport-n1000-train/checkpoints/run_csv_seed-10019_run-9_desc-pack the african in the brown box.csv.p")
    options, args = parser.parse_args()
    print(options)
    
    get_stats_for_run(options.runs_file)