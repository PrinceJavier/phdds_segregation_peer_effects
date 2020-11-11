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

def plot2d(avg_sim1, avg_sim2, avg_sim, params):

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
    kernels = params['kernels']
    thresholds = params['thresholds']   
    seed=params['seed']
    travel_lim=params['travel_lim']

    dir_ = f"charts/segregation_2d/seed_{seed}_n_{n}_kernels_{'_'.join([str(i) for i in kernels])}_props_{'_'.join([str(i) for i in props])}_thresh{'_'.join([str(i) for i in thresholds])}_travellim_{travel_lim}"
    with open(f'{dir_}/movie.gif','rb') as f:
        display(Image(data=f.read(), format='png'))


# putting them all together in a function
def segregate_2d(**params):
    
    try:
        os.mkdir('charts/segregation_2d/')
    except:
        pass
    
    # params
    n = params['n']
    agents = params['agents']
    props = params['props']
    kernels = params['kernels']
    thresholds = params['thresholds']   
    thresholds_max = params['thresholds_max'] # max thresholds if needed (list otherwise None)
    seed=params['seed']
    num_iters=params['num_iters']
    travel_lim=params['travel_lim']
    save_step=params['save_step'] 
    
    ## PART 1
    # here we break down, the algo to count neighbors per agent
    # make kernel per agent (can be different)
    k1 = kernels[0]
    k2 = kernels[1]    
    kernel_1 = np.ones(shape=(k1, k1))
    kernel_1[k1//2, k1//2] = 0 # we make the center of the kernel 0 because we don't want to count the agent in the center
    kernel_2 = np.ones(shape=(k2, k2))
    kernel_2[k2//2, k2//2] = 0
    
    # we randomly make a neighborhood based on proportions of agents and size n x n
    np.random.seed(seed) # we can adjust the seed when we're running multiple samples so we still retain replicability
    neighborhood = np.random.choice(agents, size=(n, n), replace=True, p=props) 
    
    avg_similarities_1 = [] # we track 'similarity' or 'segregation' agent 1
    avg_similarities_2 = [] # we track 'similarity' or 'segregation' agent 2
    avg_similarities_all = [] # we track 'similarity' or 'segregation'       
    
    new_dir = f"charts/segregation_2d/seed_{seed}_n_{n}_kernels_{'_'.join([str(i) for i in kernels])}_props_{'_'.join([str(i) for i in props])}_thresh{'_'.join([str(i) for i in thresholds])}_travellim_{travel_lim}"
    try:        
        os.mkdir(new_dir)        
    except:
        shutil.rmtree(new_dir)      
        os.mkdir(new_dir)          

    for iter_ in range(num_iters):

        # counting the number of neighbors using correlate2d
        # in Think Complexity, they used boundary="wrap"
        # this "tiles" the neighborhood matrix so for the cell in the top right corner, we consider the cells in the left and bottom areas
        # Here, we don't do that so we use boudary="fill" or we pad with nulls. This implies that the neighborhood has edges like in real life.
        # Note that Schelling made this same assumption that there are edges.
        # mode (counts) surrounding a cell can thus be calculated by running the "neighborhood window" or kernel from top left to bottom right.
        # mode='same' just means that we return a matrix (of counts of similar neighbors surrounding each cell) with the same shape as the neighborhood
        # the conditions are just there to tell the function whether to count or not
        agent_1 = neighborhood == 1 # say we want to know the number of agents similar to 1
        agent_2 = neighborhood == 2 # say we want to know the number of agents similar to 1

        options = dict(mode='same', boundary='fill') # here's a neat way to define arguments
        num_neighbors_1 = correlate2d(agent_1, kernel_1, **options)
        num_neighbors_2 = correlate2d(agent_2, kernel_2, **options)
        num_neighbors_all = num_neighbors_1 + num_neighbors_2 # so we exclude empty cells

        ## PART 2
        # now we find which agents are happy and which agents are not
        # we use the threshold proportions
        frac_neighbors_1 = num_neighbors_1 / num_neighbors_all
        frac_neighbors_2 = num_neighbors_2 / num_neighbors_all
        frac_neighbors_all = np.where(agent_1, frac_neighbors_1, frac_neighbors_2) # if agent_1, use agent 1 props, else agent 2
        
        # remember to just select the agent (1, or 2) and remove empty cells
        is_empty = neighborhood == 0    
        frac_neighbors_all = np.where(is_empty, np.nan, frac_neighbors_all) # then we remove empty       
        
        # making dissimiular/empty cells around an agent null
        frac_neighbors_1 = np.where(neighborhood==1, frac_neighbors_1, np.nan)
        frac_neighbors_2 = np.where(neighborhood==2, frac_neighbors_2, np.nan)       

        # getting happy agents        
        threshold_1 = thresholds[0] # threshold of agent 1
        threshold_2 = thresholds[1] # threshold of agent 2         
        
        frac_happy_1 = frac_neighbors_1 >= threshold_1
        frac_happy_2 = frac_neighbors_2 >= threshold_2

        # if we have a max threshold
        if thresholds_max != None:
            thresholds_max_1 = thresholds_max[0]
            thresholds_max_2 = thresholds_max[1]  
            frac_happy_1 = frac_happy_1 <= thresholds_max_1
            frac_happy_2 = frac_happy_2 >= thresholds_max_2    
        
        # let's be sure and remove dissimilar agents
        frac_happy_1 = np.where(neighborhood==1, frac_happy_1, np.nan)
        frac_happy_2 = np.where(neighborhood==2, frac_happy_2, np.nan)        
        
        # remove empty cells    
        frac_happy_all = np.where(agent_1, frac_happy_1, frac_happy_2) # if agent_1, use agent 1 happiness, else agent 2
        frac_happy_all = np.where(is_empty, np.nan, frac_happy_all) # then we remove empty

        # calculate the proportion of like-neighbors per agent
        avg_frac_1 = np.nanmean(frac_neighbors_1)
        avg_frac_2 = np.nanmean(frac_neighbors_2)
        avg_frac_all = np.nanmean(frac_neighbors_all) 
        
        # for tracking of segregation over time
        avg_similarities_1.append(avg_frac_1)
        avg_similarities_2.append(avg_frac_2)
        avg_similarities_all.append(avg_frac_all)        

        # PART 3
        # get the coordinates of unhappy cells
        unhappy_coords = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(frac_happy_all==0)))] # this is two lists that we need to convert to a (x, y)

        if len(unhappy_coords) ==0:
            break
        
        # get the coordinates of empty cells
        empty_coords = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(np.isnan(frac_happy_all))))]

        # switch places one agent at a time
        
        unhappy_agent_coord_ind = np.random.choice(range(len(unhappy_coords)))
        unhappy_agent_coord = unhappy_coords[unhappy_agent_coord_ind]
        unhappy_agent = neighborhood[unhappy_agent_coord]

        if travel_lim:
            k = kernels[0] # this is temporary. we assume that the kernel sizes are the same
        #     we limit the possible empty locations to only that surrounding the selected agent
        # coordinates surrounding the agent
            empty_coords = [i for i in empty_coords if i[0] >= unhappy_agent_coord[0] - k//2 
                                     and i[0] <= unhappy_agent_coord[0] + k//2 
                                     and i[1] >= unhappy_agent_coord[1] - k//2 
                                     and i[1] <= unhappy_agent_coord[1] + k//2]
            try: # if there's empty space otherwise don't move
                empty_coord_ind = np.random.choice(range(len(empty_coords)))
                empty_coord = empty_coords[empty_coord_ind]        
            except:
                empty_coord = unhappy_agent_coord # same thing
                continue # just go to the next iteration      

        else: # no travel limits    
            empty_coord_ind = np.random.choice(range(len(empty_coords)))
            empty_coord = empty_coords[empty_coord_ind]      

        # then we switch an unhappy cell with an empty cell
        neighborhood[empty_coord] = unhappy_agent
        neighborhood[unhappy_agent_coord] = 0


        # Plotting
        # save only by 10 iterations (so it's faster)
        if iter_%save_step == 0:
            f = plt.figure(figsize=(4, 4))
            plt.imshow(neighborhood)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'{new_dir}/{str(iter_).zfill(5)}.png')
            plt.close()
        
    print('all happy')        

    # save gif
    filenames = glob.glob(f"{new_dir}/*.png")
    filenames = sorted(filenames)
    filenames = filenames + [filenames[-1]] * 50 # so we freeze at the output
    images=[]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{new_dir}/movie.gif', images)
    plt.show()
    return avg_similarities_1, avg_similarities_2, avg_similarities_all
