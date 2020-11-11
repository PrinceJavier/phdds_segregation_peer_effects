# let us now prepare a function that combines all these
# let's make the random seed a parameter
# as well as whether to print or not for diagnosis

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

def plot1d(avg_sim1, avg_sim2, avg_sim, params):

    print('final avg similarities agent 1, agent 2, all ', avg_sim1[-1], avg_sim2[-1], avg_sim[-1])
    plt.plot(avg_sim1, label='1')
    plt.plot(avg_sim2, label='2')
    plt.plot(avg_sim, label='all')
    plt.xlabel('time')
    plt.ylabel('similarity between neighborhoods')
    plt.legend();

    n = params['n']
    agents = params['agents']
    props = params['props']
    k = params['k']
    thresholds = params['thresholds']   
    seed=params['seed']
    travel_lim=params['travel_lim']

    dir_ = f"charts/segregation_1d/seed_{seed}_n_{n}_k_{k}_props_{'_'.join([str(i) for i in props])}_thresh{'_'.join([str(i) for i in thresholds])}_travellim_{travel_lim}"
    with open(f'{dir_}/movie.gif','rb') as f:
        display(Image(data=f.read(), format='png'))
            


def segregate_1d(**params):
    
    try:
        os.mkdir('charts/segregation_1d/')
    except:
        pass
    
    # params
    n = params['n']
    agents = params['agents']
    props = params['props']
    k = params['k']
    thresholds = params['thresholds']   
    seed=params['seed']
    num_iters=params['num_iters']
    travel_lim=params['travel_lim']
    
    avg_similarities_1 = [] # we track 'similarity' or 'segregation' agent 1
    avg_similarities_2 = [] # we track 'similarity' or 'segregation' agent 2
    avg_similarities = [] # we track 'similarity' or 'segregation'    
    
    # we randomly make a neighborhood based on proportions of agents and size n x n
    np.random.seed(seed) # we can adjust the seed when we're running multiple samples so we still retain replicability
    neighborhood = np.random.choice(agents, size=n, replace=True, p=props) 

    # define our kernel which we use as the "radius from the agent" to get the nearby neighbors
    kernel = np.ones(shape=(k*2+1)) # it's k to the left and right of the agent
    kernel[(k*2)//2]=0 # we make the center 0 because we don't want to include the agent 

    new_dir = f"charts/segregation_1d/seed_{seed}_n_{n}_k_{k}_props_{'_'.join([str(i) for i in props])}_thresh{'_'.join([str(i) for i in thresholds])}_travellim_{travel_lim}"             
    try:        
        os.mkdir(new_dir)        
    except:
        shutil.rmtree(new_dir)      
        os.mkdir(new_dir)          

    for iter_ in range(num_iters):
        # mode='same' just means that we return a matrix (of counts of similar neighbors surrounding each cell) with the same shape as the neighborhood
        # the conditions are just there to tell the function whether to count or not
        agent_1 = neighborhood == 1 # say we want to know the number of agents similar to 1
        agent_2 = neighborhood == 2 # say we want to know the number of agents similar to 2
        num_neighbors_1 = convolve(agent_1, kernel, mode='same')
        num_neighbors_2 = convolve(agent_2, kernel, mode='same')
        num_neighbors_total = num_neighbors_1 + num_neighbors_2

        ## PART 2
        empty = neighborhood == 0

        # now we find which agents are happy and which agents are not
        happy_1a = num_neighbors_1/num_neighbors_total
        happy_2a = num_neighbors_2/num_neighbors_total
        # for both agents
        frac_same = np.where(agent_1, happy_1a, happy_2a)

        # we don't want to confuse False as both empty and unhappy
        # for all arrays, let's indicate empty cells by np.nan 
        # let's also indicate for agents 1 and 2 only those pertaining to 1 or 2
        happy_1a = np.where(neighborhood==1,np.where(empty, np.nan, happy_1a), np.nan)
        happy_2a = np.where(neighborhood==2,np.where(empty, np.nan, happy_2a), np.nan)
        frac_same = np.where(empty, np.nan, frac_same)

        # get the proportion of same neighbors in locality of each agent
        avg_similarity_1 = np.nanmean(happy_1a)
        avg_similarity_2 = np.nanmean(happy_2a)
        avg_similarity = np.nanmean(frac_same) # proportion of same neighbots

        avg_similarities_1.append(avg_similarity_1)
        avg_similarities_2.append(avg_similarity_2)
        avg_similarities.append(avg_similarity)  

        # how many happy
        threshold_1 = thresholds[1] # threshold for agent 1
        happy_1 = happy_1a > threshold_1
        threshold_2 = thresholds[2] # threshold for agent 2
        happy_2 = happy_2a > threshold_2

        # we just want one array
        # get the value from happy_1 if the agent =1 , else get happy_2
        happy_all = np.where(agent_1, happy_1, happy_2)
        happy_all = np.where(empty, np.nan, happy_all)

        ## PART 3
        # from left to right (or randomly), we get the locations of unhappy agents
        # and place them into an empty cell that makes the agent happy (from left to right or randomly)
        # otherwise, randomly place

        # first look for the location of an unhappy agent
        sources = np.argwhere(happy_all==0).flatten() # indices of possible sources
        try:        
            # # if from left to right            
            # source = sources[0]        
            # if random       
            source = np.random.choice(sources, size=1)[0]
        except:
            break

        current_agent = neighborhood[source] # note that the agent taggings must = index [0, 1, 2] otherwise pipe will fail
        agent_ = neighborhood == current_agent # we get similar neighbors
        num_neighbors_ = convolve(agent_, kernel, mode='same') # get the number of neighbors per cell
        threshold_ = thresholds[current_agent] # we use current agent as index of threshold
        happy_ = num_neighbors_/num_neighbors_total > threshold_

        # only get those with empty
        empty = neighborhood == 0
        happy_ = np.where(empty, happy_, np.nan) # if empty, we get it

        # with travel limits
        if travel_lim:
            dests = np.argwhere(happy_==True).flatten() # indices of possible destinations
            dests = [i for i in dests if i <= source+k and i >= source-k]# get destinations that are k away from source
            try: # if we can find a destination 
                dest = np.random.choice(dests, size=1)[0]
            except: # randomly assign within travel limit
                empty_inds = np.argwhere(empty==True).flatten() # indices of empty spots                    
                empty_inds = [i for i in empty_inds if i <= source+k and i >= source-k]# get empty points that are k away from source                                    
                try:
                    dest = np.random.choice(empty_inds)
                except:
                    dest = source # if we can't find a vacant spot, we don't move
        # without travel limits
        else:
            dests = np.argwhere(happy_==True).flatten() # indices of possible destinations
            try: # if we can find a destination
                # dest = dests[0] # we get the first one (or we can get randomly)
                dest = np.random.choice(dests, size=1)[0]
            except: # randomly assign       
                empty_inds = np.argwhere(empty==True).flatten() # indices of empty spots    
                try:
                    dest = np.random.choice(empty_inds)
                except:
                    dest = source # if we can't find a vacant spot, we don't move
                    continue # since we don't really change things
        # we now move the agent from the destination to the source
        # we then do this for several iterations
        neighborhood[dest] = current_agent
        neighborhood[source] = 0 # empty now          

        # Plotting
        f = plt.figure(figsize=(15, 2))
        plt.imshow(neighborhood.reshape(-1, 1).T)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{new_dir}/{str(iter_).zfill(3)}.png')
        plt.close()

    # save gif
    filenames = glob.glob(f"{new_dir}/*.png")
    filenames = sorted(filenames)
    filenames = filenames + [filenames[-1]] * 15 # so we freeze at the output
    images=[]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{new_dir}/movie.gif', images)
    
    plt.show()
    return avg_similarities_1, avg_similarities_2, avg_similarities
        
   