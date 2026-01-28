from os import path
import argparse
import parse_table1
import parse_parameter_eval_table
import matplotlib.pyplot as plt

plot_colors_teams = {
        1: '#F5931B', # Orange
        2: '#418A04', # Green
        3: '#04608A', # Light navy
        4: '#71048A' # Purple
    }
plot_colors_class = {
        'PSPL': '#F5931B', # Orange
        'Binary_star': '#71048A', # Purple
        'Binary_planet': '#04608A' # Light navy
    }
plot_symbols_class = {
        'PSPL': '^',
        'Binary_star': '*',
        'Binary_planet': 's'
    }

def analyse_unmodeled_events(args):

    # Load the master table of the simulated parameter values
    master_data = parse_table1.read_master_table(args.master_file)

    # Load the parameter evaluation table for each team, compare it with the master_data
    # table and identify and record any missing event models
    team_results = {}
    for teamID in range(1,5,1):
        file_path = getattr(args, 'eval_table' + str(teamID))
        eval_table = parse_parameter_eval_table.load_parameter_evaluation_table(file_path)

        # For each team, compare the results against the master table and identify the
        # events for which no results were produced
        team_results[teamID] = find_unmodeled_events(master_data, eval_table, teamID)

        # Output machine-readable log of events missed by each team
        record_missed_events(args, master_data, team_results, teamID)

    # Summarize the parameters of the unmodeled events for each team
    # Explore the distributions of tE, t0, u0, model_class, and for binary events,
    # s, q and M2
    plot_twoparameter_distributions(args, team_results, master_data, 'u0', 'tE', '$u_{0}$', '$t_{E}$ [days]')
    plot_twoparameter_distributions(args, team_results, master_data, 's', 'q', 'Proj. separation', 'Mass ratio')
    plot_twoparameter_distributions(args, team_results, master_data, 't0', 'tE', '$t_{0}$', '$t_{E}$ [days]')
    plot_twoparameter_distributions(args, team_results, master_data, 'piE', 'rho', '$\pi_{E}$', '$\\rho$')
    plot_twoparameter_distributions(args, team_results, master_data, 'tE', 'piE', '$t_{E}$', '$\pi_{E}$')

    # Compare the lists of unmodeled events to look for events that were missed by multiple teams
    compare_missed_lists(args, team_results)

def compare_missed_lists(args, team_results):
    """
    Function to compare the lists of missed events to look for events that were missed by multiple teams
    """

    # Compile a full list of all missed events across all teams
    all_missed_events = []
    for teamID in range(1, 5, 1):
        all_missed_events += [eventID for eventID in team_results[teamID] if eventID not in all_missed_events]
    print('Combined, all teams missed a total of ' + str(len(all_missed_events)) + ' events')

    # Create a table for each of the missed events identifying which teams missed it
    miss_rates = {1: 0, 2: 0, 3: 0, 4: 0}
    file_path = path.join(args.data_dir, 'missed_events_table.txt')
    with open(file_path, 'w') as f:
        f.write('# EventID  Team1   Team2   Team3   Team4  missed_by\n')

        for eventID in all_missed_events:
            output = eventID
            nmissed = 0
            for teamID in range(1, 5, 1):
                if eventID in team_results[teamID]:
                    output += ' 0'
                    nmissed +=1
                else:
                    output += ' 1'
            output += ' ' + str(nmissed)
            f.write(output + '\n')
            miss_rates[nmissed] += 1

    for rate, nevents in miss_rates.items():
        print(str(nevents) + ' were missed by ' + str(rate) + ' team(s)')

def plot_twoparameter_distributions(args, team_results, master_data, xpar, ypar, xlabel, ylabel):
    """
    Function to plot parameter distributions of the missed events for each team
    """

    ncol = 2
    nrow = 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(12,10), layout='constrained')

    irow = 0
    icol = 0
    for teamID in range(1, 5, 1):
        handles = []
        labels = []

        # Plot the true distribution in the background for comparison
        par1 = [getattr(event, xpar) for eventID, event in master_data.items()]
        par2 = [getattr(event, ypar) for eventID, event in master_data.items()]
        obj, = axs[irow,icol].plot(
            par1, par2,
            marker='o',
            mfc='grey', mec='grey',
            ls='none',
            alpha=0.3
        )
        if 'Simulation' not in labels:
            handles.append(obj)
            labels.append('Simulation')

        # Overplot the parameters for the events missed by each team
        par1 = [getattr(master_data[eventID], xpar) for eventID in team_results[teamID]]
        par2 = [getattr(master_data[eventID], ypar) for eventID in team_results[teamID]]
        colors = [plot_colors_class[master_data[eventID].model_class] for eventID in team_results[teamID]]
        symbols = [plot_symbols_class[master_data[eventID].model_class] for eventID in team_results[teamID]]
        models = [master_data[eventID].model_class for eventID in team_results[teamID]]

        for j in range(0,len(par1),1):
            if par1[j] and par2[j]:
                obj, = axs[irow,icol].plot(
                    par1[j], par2[j],
                    marker=symbols[j],
                    mfc=colors[j], mec=colors[j],
                    ls='none',
                    alpha=1.0)
                if models[j] not in labels:
                    handles.append(obj)
                    labels.append(models[j])

        axs[irow,icol].set_xlabel(xlabel, fontsize=18)
        axs[irow,icol].set_ylabel(ylabel, fontsize=18)
        axs[irow,icol].tick_params(axis='x', labelsize=16)
        axs[irow,icol].tick_params(axis='y', labelsize=16)

        axs[irow,icol].set_title('Team ' + str(teamID), fontsize=18)
        axs[irow,icol].grid()

        icol += 1
        if icol == ncol:
            icol = 0
            irow +=1

    axs[1, 1].legend(
        handles=handles,
        labels=labels,
        ncol=2,
        bbox_to_anchor=(1.05, -0.2),
        fontsize=16
    )

    plt.subplots_adjust(left=0.02, bottom=0.01, right=0.95, top=0.95, wspace=0.3, hspace=0.3)

    plt.tight_layout()
    plt.savefig(path.join(args.data_dir, xpar + ypar + '_distribution.png'))

def find_unmodeled_events(master_data, eval_table, teamID):
    """
    Function to identify any events in the master table for which no valid model is
    present in the model parameter's table.  Note that different teams may have handled
    unmodeled events differently, so here we look both for missing event IDs and events
    with invalid t0, tE and u0 values.
    Note also that since the challenge dataset included variable stars, an event only
    counts as missed if the true model class is microlensing

    Parameters:
        master_data dict  Input table of simulated event parameters
        eval_table dict   Modeled event parameters from a single team

    Returns:
        missed_events  list  Set of EventEntry objects for unmodeled events
    """

    missed_events = []
    event_classes = ['PSPL', 'Binary_star', 'Binary_planet']

    for eventID, true_params in master_data.items():
        if true_params.model_class in event_classes:
            # Check whether this eventID is present in the parameter table
            if eventID not in eval_table.keys():
                missed_events.append(eventID)

            # If the event is in the parameter table, check the results include
            # valid floating point values for the t0, tE and u0.  Note we don't check
            # whether these are correct since that was part of a different evaluation.
            # Here we are just looking for events that couldn't be modelled at all.
            else:
                model_params = eval_table[eventID]
                if any(['none' in str(x).lower() for x in [model_params.t0, model_params.tE, model_params.u0]]):
                    missed_events.append(eventID)

    print('Team ' + str(teamID) + ' found ' + str(len(missed_events)) + ' missed events')

    return missed_events

def record_missed_events(args, master_data, team_results, teamID):
    """
    Function to output a machine readable log of the missed events for each team
    """

    file_path = path.join(args.data_dir, 'missed_events_team' + str(teamID) + '.txt')
    with open(file_path, 'w') as f:
        for i,eventID in enumerate(team_results[teamID]):
            event = master_data[eventID]

            if i == 0:
                header = ' '.join([par[0] for par in event.requirements])
                f.write('# ' + header + '\n')
            f.write(event.summarize_parameters() + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('master_file', help='Path to the master table file')
    parser.add_argument('eval_table1', help='Path to Team 1s evaluation table')
    parser.add_argument('eval_table2', help='Path to Team 2s evaluation table')
    parser.add_argument('eval_table3', help='Path to Team 3s evaluation table')
    parser.add_argument('eval_table4', help='Path to Team 4s evaluation table')
    parser.add_argument('data_dir', help='Path to output directory')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    analyse_unmodeled_events(args)