import pickle
import numpy as np
import os
from optparse import OptionParser
import scipy.stats as st
import scipy.spatial
import matplotlib.pyplot as plt

def bar_plot(labels, values, std_errs, save_path, y_label, title):
    fig, ax = plt.subplots()
    
    x_pos=np.array(list(range(values.shape[0])))
    
    ax.bar(x_pos, values, yerr=values-std_errs[:,0], align='center', alpha=0.5, ecolor='black', capsize=7)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
#     ax.yaxis.grid(True)
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title}_{y_label}.png'))
#     plt.show()

def get_stats_for_run(runs_file):
    ''' Print out averages per identity per command.

    args:
        runs_file: string, either run log or folder containing run logs.

    '''

    save_path=runs_file+"_plots/"
    os.mkdir(save_path)
    if os.path.isdir(runs_file):
        for file in os.listdir(runs_file):
            all_runs= []
            if file[-2:] == '.p':
                runs=pickle.load(open(os.path.join(runs_file, file), 'rb'))
                all_runs += runs
    else:
        all_runs=pickle.load(open(runs_file, 'rb')) 

    # Parse data list into dict
    #Data dictionary, cmd_str: run num: step_num: step info
    data_dict={}
    for run in all_runs:
        cmd_str=run[2]
        if cmd_str not in data_dict:
            data_dict[cmd_str]={}
        run_num=run[1]
        if run_num not in data_dict[cmd_str]:
            data_dict[cmd_str][run_num]={}
        step_num=run[3]
        # Run is a list of information about each step as outlined in the powerpoint
        data_dict[cmd_str][run_num][step_num]=run

    # Compute placed rankings and frequencies for each cmd and identity
    # cmd_str: ethnicity: gender: order object placed in
    rankings_dict={}
    # cmd_str: ethnicity: gender: whether object was placed
    placed_dict={}
    # cmd_str: ethnicity: gender: whether object was moved more than 1mm
    moved_dict={}
    # for each cmd_str
    for cmd in data_dict:
        if cmd not in rankings_dict:
            rankings_dict[cmd]={}
            placed_dict[cmd]={}
            moved_dict[cmd]={}
        # of each run
        for run_num in data_dict[cmd]:
            # order each step in run
            try:
                ordered_runs=[data_dict[cmd][run_num][i] for i in range(len(data_dict[cmd][run_num]))]
            except KeyError:
                print("key not found")
                continue       
            ids=[]
            raw_order=[]
            # for each object
            for obj_ind in range(4, len(ordered_runs[0]), 3):
                #compute metrics for that object
                identity=ordered_runs[0][obj_ind][0]
                
                # Compute if an object was placed at all
                placed_status=np.array([ordered_runs[i][obj_ind+2] for i in range(len(ordered_runs))])
                placed=np.sum(placed_status)>0
                
                # Expand dicts as needed
                if identity[0] not in placed_dict[cmd]:
                    placed_dict[cmd][identity[0]]={}
                    rankings_dict[cmd][identity[0]]={}
                    moved_dict[cmd][identity[0]]={}
                if identity[1] not in placed_dict[cmd][identity[0]]:
                    placed_dict[cmd][identity[0]][identity[1]]=[]
                    rankings_dict[cmd][identity[0]][identity[1]]=[]
                    moved_dict[cmd][identity[0]][identity[1]]=[]
                    
                placed_dict[cmd][identity[0]][identity[1]].append(placed)
                ids.append(identity)
                
                #compute whether object moved
                positions=np.array([ordered_runs[i][obj_ind+1][0] for i in range(len(ordered_runs))])
                dists=scipy.spatial.distance.cdist(positions, positions)
                moved=np.amax(dists)>1e-3
                moved_dict[cmd][identity[0]][identity[1]].append(moved)
                
                # If object was placed, compute step it was placed at
                if placed==1:
                    raw_order.append(np.argwhere(placed_status)[0,0])
                # If not, say it was placed at last step
                else:
                    raw_order.append(placed_status.shape[0])
            
            # Compute *relative* order objects were placed in
            ordering=np.argsort(np.array(raw_order))
            for ind in range(ordering.shape[0]):
                if raw_order[ind]==placed_status.shape[0]:
                    order=placed_status.shape[0]
                else:
                    order=ordering[ind]
                identity=ids[ind]
                rankings_dict[cmd][identity[0]][identity[1]].append(order)
            u=0
    
    # Tuple of dicts with numerical values for each identity
    dicts=(rankings_dict, placed_dict, moved_dict)
    # Names of each metric
    metric_names=("order object placed", "object placed", "object moved")
    # Compute means and 90% CIs for each identity-metric dict
    
    for d_ind in range(len(dicts)):
        all_values={}
        data_dict=dicts[d_ind]
        for cmd in data_dict:
            id_labels=[]
            means=[]
            stds=[]
            
            # ethnicity x gender
            for id_1 in data_dict[cmd]:
                for id_2 in data_dict[cmd][id_1]:
                    data=np.array(data_dict[cmd][id_1][id_2])
                    # Compute metric mean
                    mean=np.mean(data)
                    # Compute 90% confidence interval
                    low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                    high_err=mean+(mean-low_err)
                    
                    id_labels.append(id_1+id_2)
                    means.append(mean)
                    stds.append([low_err, high_err])
                    
                    if id_labels[-1] not in all_values:
                        all_values[id_labels[-1]]=[]
                    all_values[id_labels[-1]].append(data)
                    
                    print(f"{cmd} | {metric_names[d_ind]} | {id_1} | {id_2} | mean: {mean} CI: ({low_err}, {high_err})")
            
            # ethnicity     
            for id_1 in data_dict[cmd]:
                data=[]
                for id_2 in data_dict[cmd][id_1]:
                    data.append(data_dict[cmd][id_1][id_2])
                data=np.array(data).reshape(-1)
                # Compute metric mean
                mean=np.mean(data)
                # Compute 90% confidence interval
                low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                high_err=mean+(mean-low_err)
                
                id_labels.append(id_1)
                means.append(mean)
                stds.append([low_err, high_err])
                
                if id_labels[-1] not in all_values:
                    all_values[id_labels[-1]]=[]
                all_values[id_labels[-1]].append(data)
                
                print(f"{cmd} | {metric_names[d_ind]} | {id_1} | mean: {mean} CI: ({low_err}, {high_err})")
            
            # gender     
            for id_2 in data_dict[cmd][list(data_dict[cmd].keys())[0]]:
                data=[]
                for id_1 in data_dict[cmd]:
                    data.append(data_dict[cmd][id_1][id_2])
                data=np.array(data).reshape(-1)
                # Compute metric mean
                mean=np.mean(data)
                # Compute 90% confidence interval
                low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
                high_err=mean+(mean-low_err)
                
                id_labels.append(id_2)
                means.append(mean)
                stds.append([low_err, high_err])
                
                if id_labels[-1] not in all_values:
                    all_values[id_labels[-1]]=[]
                all_values[id_labels[-1]].append(data)
                
                print(f"{cmd} | {metric_names[d_ind]} | {id_2} | mean: {mean} CI: ({low_err}, {high_err})")
            
            means=np.array(means)
            stds=np.array(stds)
            
            # Plot results for specific command
            bar_plot(id_labels, means, stds, save_path, metric_names[d_ind], cmd)
        
        # Plot results for all commands
        # ethnicity x gender for all cmds
        all_means=[]
        all_ids=[]
        all_stds=[]
        for id in all_values:
            data=np.concatenate(all_values[id])
            # Compute metric mean
            mean=np.mean(data)
            # Compute 90% confidence interval
            low_err=st.t.interval(0.9, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]
            high_err=mean+(mean-low_err)
            
            all_ids.append(id)
            all_means.append(mean)
            all_stds.append([low_err, high_err])
            
        all_means=np.array(all_means)
        all_stds=np.array(all_stds)
        bar_plot(all_ids, all_means, all_stds, save_path, metric_names[d_ind], "All Commands")

    

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--runs_file", dest="runs_file", default="/home/willie/github/cliport/cliport_quickstart/packing-unseen-google-objects-race-seq-cliport-n1000-train/checkpoints/packing-unseen-google-objects-race-seq-cliport-n100-multi-results-test.json_two.p")
    options, args = parser.parse_args()
    print(options)

    get_stats_for_run(options.runs_file)