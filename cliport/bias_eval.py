import pickle
import numpy as np
import os
import pandas as pd
from optparse import OptionParser
import scipy.stats as st
import scipy.spatial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import csv

def tukey_test(data, save_path, title):
    ''' Run pairwise Tukey test to determine p-values for differences between data means.
    
    args:
        data: dict {identity: list of samples}
        save_path: str, where to save csv of results
        title: str, name of csv
    '''
    
    identities=[]
    per_data_identities=[]
    datas=[]
    for id in data:
        identities.append(id)
        for _ in range(data[id].shape[0]):
            per_data_identities.append(id)
        datas.append(data[id])
    
    # Can't do multiple comparison tests with only one group
    
    anova_oneway=f_oneway(*datas)

    # perform Tukey's test
    flat_datas=np.concatenate(datas)
    try:
        tukey = pairwise_tukeyhsd(endog=flat_datas,
                                  groups=per_data_identities,
                                  alpha=0.1)
    except ValueError:
        return
    u=print(tukey)
    
    results=[["anova statistic", anova_oneway.statistic, anova_oneway.pvalue]]
    for row in tukey._results_table:
        results.append([])
        for data in row.data:
            results[-1].append(str(data))
    
    with open(os.path.join(save_path, title+".csv"), "w") as csvfile:
        csv_writer=csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(results)
        
    
    u=0
    
    

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

def get_stats_for_run(runs_file, cmd_subsets, subset_names):
    ''' Print out averages per identity per command.

    args:
        runs_file: string, either run log or folder containing run logs.

    '''

    save_path=runs_file+"_plots/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.isdir(runs_file):
        all_runs= []
        for file in os.listdir(runs_file):
            if file[-2:] == '.p':
                file_to_load = os.path.join(runs_file, file)
                print('loading: ' + file_to_load)
                runs=pickle.load(open(file_to_load, 'rb'))
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
                print(f"key not found {cmd} {run_num} {len(data_dict[cmd][run_num])} runs found")
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
#                 # If not, say it was placed at last step
#                 else:
#                     raw_order.append(placed_status.shape[0])
            
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
    for cmd_subset_ind in range(len(cmd_subsets)):
        cmd_list=cmd_subsets[cmd_subset_ind]
        subset_name=subset_names[cmd_subset_ind]
        
        if len(subset_name)>0:
            cmd_save_path=os.path.join(save_path, subset_name)
            if not os.path.exists(cmd_save_path):
                os.mkdir(cmd_save_path)
        else:
            cmd_save_path=save_path
        
        for d_ind in range(len(dicts)):
            all_values={}
            data_dict=dicts[d_ind]
            for cmd in data_dict:
                if cmd in cmd_list or len(cmd_list)==0:
                    id_labels=[]
                    means=[]
                    stds=[]
                    
                    # ethnicity x gender
                    # dict of data aggregated by ethnicity|gender
                    cmd_data_dict={}
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
                            cmd_data_dict[id_labels[-1]]=data
                            
                            print(f"{cmd} | {metric_names[d_ind]} | {id_1} | {id_2} | mean: {mean} CI: ({low_err}, {high_err})")
                    tukey_test(cmd_data_dict, cmd_save_path, f'tukey_test_{cmd}_ethnicityxgender')
                    
                    # ethnicity
                    # dict of data aggregated by ethnicity
                    cmd_data_dict={} 
                    for id_1 in data_dict[cmd]:
                        data=[]
                        for id_2 in data_dict[cmd][id_1]:
                            data.append(data_dict[cmd][id_1][id_2])
                        data=np.concatenate(data)
                        cmd_data_dict[id_1]=data
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
                    tukey_test(cmd_data_dict, cmd_save_path, f'tukey_test_{cmd}_ethnicity')
                    
                    # gender
                    # dict of data aggregated by gender
                    cmd_data_dict={}    
                    for id_2 in data_dict[cmd][list(data_dict[cmd].keys())[0]]:
                        data=[]
                        for id_1 in data_dict[cmd]:
                            data.append(data_dict[cmd][id_1][id_2])
                        data=np.concatenate(data)
                        cmd_data_dict[id_2]=data
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
                    tukey_test(cmd_data_dict, cmd_save_path, f'tukey_test_{cmd}_gender')
                    
                    means=np.array(means)
                    stds=np.array(stds)
                    
                    # Plot results for specific command
                    bar_plot(id_labels, means, stds, cmd_save_path, metric_names[d_ind], cmd)
                    
                
            # Plot results for all commands
            # ethnicity x gender for all cmds
            all_means=[]
            all_ids=[]
            all_stds=[]
            all_data_dict_ethnicity_gender={}
            all_data_dict_ethnicity={}
            all_data_dict_gender={}
            for id in all_values:
                data=np.concatenate(all_values[id])
                
                if id in ["M", "F"]:
                    all_data_dict_gender[id]=data
                elif id in ["A", "B", "L", "W"]:
                    all_data_dict_ethnicity[id]=data
                else:
                    all_data_dict_ethnicity_gender[id]=data
                
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
            bar_plot(all_ids, all_means, all_stds, cmd_save_path, metric_names[d_ind], "All Commands")
            
            tukey_test(all_data_dict_ethnicity_gender, cmd_save_path, f'tukey_test_all_cmds_ethnicity_gender')
            tukey_test(all_data_dict_ethnicity, cmd_save_path, f'tukey_test_all_cmds_ethnicity')
            tukey_test(all_data_dict_gender, cmd_save_path, f'tukey_test_all_cmds_gender')

    

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--runs_file", dest="runs_file", default="/home/willie/github/cliport/cliport_quickstart/packing-unseen-google-objects-race-seq-cliport-n1000-train/checkpoints/checkpoints_test_cfd-180-strings-2022-01-11-1218 2/checkpoints_test_cfd-180-strings-2022-01-11-1218/checkpoints/")
    options, args = parser.parse_args()
    print(options)

    get_stats_for_run(options.runs_file, [["pack the man in the brown box"], []], ["lawyer", "all"])
    
    