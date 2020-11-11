# putting them all together

# since we now know the fraction of like-agents at any one time,
# we can define a threshold distribution per agent and that will define whether they will move out or move in
# we need a way to track if an individual with a certain threshold is inside our outside the sub-neighborhood
# we initialize by randomly assigning whether an agent with a threshold is inside our outside with 0s or 1s

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import convolve # for counting neighbors in 1d
from scipy.signal import correlate2d # for counting neighbors in 2d
import os, shutil
import imageio
import glob
from tqdm import tqdm
from IPython.display import Image
from skimage.transform import resize

# define a distribution
def thresh_dist(x):
    # accepts an array and applies a function
    # must be between 0 and 1
    return -x/np.max(x)+1 # decreasing slope (0 is most tolerant, last user is least toleratn)


def plot_bounded(avg_sim1, avg_sim2, params):
    print('final avg similarities agent 1, agent 2 ', avg_sim1[-1], avg_sim2[-1])
    plt.plot(avg_sim1, label='1')
    plt.plot(avg_sim2, label='2')
    plt.xlabel('time')
    plt.ylabel('similarity between neighborhoods')
    plt.legend();

    n = params['n']
    props = params['props']
    seed=params['seed']
    
    dir_ = f"charts/segregation_bounded/seed_{seed}_n_{n}_props_{'_'.join([str(i) for i in props])}"
    with open(f'{dir_}/movie.gif','rb') as f:
        display(Image(data=f.read(), format='png'))


def segregate_bounded(**params):
    
    try:
        os.mkdir('charts/segregation_bounded/')
    except:
        pass
    
    # PARAMS
    n = params['n']
    pattern=params['pattern'] # map pattern
    agents = params['agents']
    props = params['props']
    seed=params['seed']
    num_iters=params['num_iters']
    save_step=params['save_step']

    new_dir = f"charts/segregation_bounded/seed_{seed}_n_{n}_props_{'_'.join([str(i) for i in props])}"

    try:        
        os.mkdir(new_dir)        
    except:
        shutil.rmtree(new_dir)      
        os.mkdir(new_dir)      
    
    avg_similarities_1 = [] # similarities for agent 1 inside neighborhood
    avg_similarities_2 = [] # similarities for agent 2 inside neighborhood

    # INITIALIZE
    # we randomly make a neighborhood based on proportions of agents and size n x n
    np.random.seed(seed) # we can adjust the seed when we're running multiple samples so we still retain replicability
    neighborhood = np.random.choice(agents, size=(n, n), replace=True, p=props) 

    # get the actual size neighborhood "map" which tells us which is inside our outside the sub-neighborhood
    neighborhood_map = resize(pattern, (n, n), anti_aliasing=False, preserve_range=True)
    neighborhood_map = np.round(neighborhood_map)

    # PART 1
    # Define distributions
    # use external threshold function
    # get the agents for the entire neighborhood
    agent_1 = np.where(neighborhood == 1, neighborhood, 0)  # say we want to know the number of agents similar to 1
    agent_2 = np.where(neighborhood == 2, neighborhood, 0) # say we want to know the number of agents similar to 1
    thresh_agent_1 = thresh_dist(np.array(range(int(np.nansum(agent_1))))) # index is the agent, value is threshold
    thresh_agent_2 = thresh_dist(np.array(range(int(np.nansum(agent_2))))) # index is the agent, value is threshold
    
    for iter_ in range(num_iters):
        # PART 2
        # get the agents for the entire neighborhood
        # make empty spaces nan
        agent_1 = np.where(neighborhood == 1, neighborhood, 0)  # say we want to know the number of agents similar to 1
        agent_2 = np.where(neighborhood == 2, neighborhood, 0) # say we want to know the number of agents similar to 1
        agent_all = neighborhood != 0
        
        # only get the values inside the sub-neighborhood via element-wise multiplication with the pattern
        sub_neighborhood = np.multiply(neighborhood_map, neighborhood)
        # make outside neighborhood nulls
        sub_neighborhood = np.where(neighborhood_map==0, np.nan, sub_neighborhood)
        # get outside of neighborhood
        out_neighborhood = np.where(neighborhood_map==1, np.nan, neighborhood)

        # get the agents inside the subneighborhood
        agent_1_sub = np.where(neighborhood_map==1, agent_1==1, np.nan)
        agent_2_sub = np.where(neighborhood_map==1, agent_2==2, np.nan)
        agent_all_sub = np.where(neighborhood_map==1, agent_all!=0, np.nan)

        # # get the agents outside the subneighborhood
        agent_1_out = np.where(neighborhood_map==0, agent_1==1, np.nan)
        agent_2_out = np.where(neighborhood_map==0, agent_2==2, np.nan)
        agent_all_out = np.where(neighborhood_map==0, agent_all, np.nan)

        # we can compute the fraction of similar agents / total agents in the neighborhood
        agent_1_frac = np.nansum(agent_1_sub)/np.nansum(agent_all_sub)
        agent_2_frac = np.nansum(agent_2_sub)/np.nansum(agent_all_sub)

        avg_similarities_1.append(agent_1_frac)
        avg_similarities_2.append(agent_2_frac)        
        

        # PART 3
        # Initialize monitoring if agent with certain threshold is inside our outside the subneighborhoood
        if iter_ == 0:
            # number of agents inside and outside
            num_inside_1 = int(np.nansum(agent_1_sub))
            num_inside_2 = int(np.nansum(agent_2_sub))
            num_outside_1 = int(np.nansum(agent_1) - num_inside_1)
            num_outside_2 = int(np.nansum(agent_2) - num_inside_2)

            # threshold locs - > we use this to check which of the list of thresholds (agents) are inside (1) or outside (0)
            # we don't need a specific agent tied to a specific threshold
            # we just need to pick an agent inside/outside and then indicate in the threshold that it's now 1 or 0
            # we only need to initialize once
            thresh_locs_1 = np.array(list(np.ones(shape=(num_inside_1))) + list(np.zeros(shape=(num_outside_1))))
            thresh_locs_2 = np.array(list(np.ones(shape=(num_inside_2))) + list(np.zeros(shape=(num_outside_2))))
            np.random.shuffle(thresh_locs_1)
            np.random.shuffle(thresh_locs_2)

        # PART 4
        # get indices of those inside which we will use to map back to actual thresholds
        # the index of a threshold that's breached will then be used to change the mapping of whether inside or outside in thresh_locs_1 or 2
        inside_inds_1 = np.nonzero(thresh_locs_1)
        inside_inds_2 = np.nonzero(thresh_locs_2)

        # get indices of those outside
        outside_inds_1 = np.nonzero(1-thresh_locs_1)
        outside_inds_2 = np.nonzero(1-thresh_locs_2)

        # we put these here and run as a group because it's easier to track
        # we get the thresholds for those inside and check which ones of them (from highest to lowest) will need to get out
        highest_ind_inside_1 = np.argsort(thresh_agent_1[inside_inds_1])[0]
        highest_thresh_inside_1 = thresh_agent_1[highest_ind_inside_1]

        highest_ind_inside_2 = np.argsort(thresh_agent_2[inside_inds_2])[0]
        highest_thresh_inside_2 = thresh_agent_2[highest_ind_inside_2]

        # we get the thresholds for those outside and check from lowest to highest which ones of them will go in
        lowest_ind_outside_1 = np.argsort(thresh_agent_1[outside_inds_1])[0]
        lowest_thresh_outside_1 = thresh_agent_1[lowest_ind_outside_1]

        lowest_ind_outside_2 = np.argsort(thresh_agent_2[outside_inds_2])[0]
        lowest_thresh_outside_2 = thresh_agent_2[lowest_ind_outside_2]

        # we randomly select between four options: check inside vs outside and agent 1 vs agent 2 like rolling a die
        area_to_check = np.random.choice(['inside', 'outside'])
        agent_to_check = np.random.choice([1, 2])
        
#         print(area_to_check, agent_to_check)

        if area_to_check=='inside' and agent_to_check == 1:
            if highest_thresh_inside_1 >= agent_1_frac: # this agent leaves if the proportion needed is higher than actual
                # we randomly select one agent from the subneighborhood and randomly place in a vacant area outside
                sources = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_1_sub==1)))] # this is two lists that we need to convert to a (x, y)        
                
                if len(sources)==0:
                    continue                
                
                source_ind = np.random.choice(range(len(sources)))
                source = sources[source_ind]

                # randomly get empty coordinates outside the sub-neighborhood
                dests = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_all_out==0)))]
                if len(dests) == 0:
                    break                
                dest_ind = np.random.choice(range(len(dests)))        
                dest = dests[dest_ind]

                # switch cells
                neighborhood[dest] = 1              
                neighborhood[source] = 0        

                # we update the agent-threshold location tracker thresh_locs_1
                thresh_locs_1[highest_ind_inside_1] = 0 # now outside

        if area_to_check=='inside' and agent_to_check == 2:
            if highest_thresh_inside_2 >= agent_2_frac: # this agent leaves if the proportion needed is higher than actual
                # we randomly select one agent from the subneighborhood and randomly place in a vacant area outside
                sources = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_2_sub==1)))] # this is two lists that we need to convert to a (x, y)        
                
                if len(sources)==0:
                    continue
                
                source_ind = np.random.choice(range(len(sources)))
                source = sources[source_ind]

                # randomly get empty coordinates outside the sub-neighborhood
                dests = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_all_out==0)))]
                if len(dests) == 0:
                    break
                dest_ind = np.random.choice(range(len(dests)))        
                dest = dests[dest_ind]

                # switch cells
                neighborhood[dest] = 2                    
                neighborhood[source] = 0        

                # we update the agent-threshold location tracker thresh_locs_1
                thresh_locs_2[highest_ind_inside_2] = 0 # now outside   

        if area_to_check=='outside' and agent_to_check == 1:
            if lowest_thresh_outside_1 <= agent_1_frac: # this agent enters if the proportion needed is lower than actual
                # we randomly select one agent from the subneighborhood and randomly place in a vacant area outside
                sources = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_1_out==1)))] # this is two lists that we need to convert to a (x, y)        
                
                if len(sources)==0:
                    continue
                
                source_ind = np.random.choice(range(len(sources)))
                source = sources[source_ind]

                # randomly get empty coordinates inside the sub-neighborhood
                dests = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_all_sub==0)))] # empty space inside
                if len(dests) == 0:
                    break
                dest_ind = np.random.choice(range(len(dests)))        
                dest = dests[dest_ind]

                # switch cells
                neighborhood[dest] = 1           
                neighborhood[source] = 0        

                # we update the agent-threshold location tracker thresh_locs_1
                thresh_locs_1[lowest_ind_outside_1] = 1 # now outside 

        if area_to_check=='outside' and agent_to_check == 2:
            if lowest_thresh_outside_2 <= agent_2_frac: # this agent enters if the proportion needed is lower than actual
                # we randomly select one agent from the subneighborhood and randomly place in a vacant area outside
                sources = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_2_out==1)))] # this is two lists that we need to convert to a (x, y)        

                if len(sources)==0:
                    continue               
                
                source_ind = np.random.choice(range(len(sources)))
                source = sources[source_ind]

                # randomly get empty coordinates outside the sub-neighborhood
                dests = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(agent_all_sub==0)))] # empty space inside
                if len(dests) == 0:
                    break
                dest_ind = np.random.choice(range(len(dests)))        
                dest = dests[dest_ind]

                # switch cells
                neighborhood[dest] = 2                 
                neighborhood[source] = 0        

                # we update the agent-threshold location tracker thresh_locs_1
                thresh_locs_2[lowest_ind_outside_2] = 1 # now outside      
                
        # Plotting
        # save only by n iterations (so it's faster)
        if iter_%save_step == 0:
            f = plt.figure(figsize=(4, 4))
            plt.imshow(neighborhood)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'{new_dir}/{str(iter_).zfill(5)}.png')
            plt.close()
        
    print('complete')        

    # save gif
    filenames = glob.glob(f"{new_dir}/*.png")
    filenames = sorted(filenames)
    filenames = filenames + [filenames[-1]] * 50 # so we freeze at the output
    images=[]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{new_dir}/movie.gif', images)
    plt.show()
    return avg_similarities_1, avg_similarities_2                
