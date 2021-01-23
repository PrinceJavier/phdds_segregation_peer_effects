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

class segregate_1d:    
    def __init__(self, n = 70, props = [0.2, 0.4, 0.4], k = 4, 
                 thresholds = [0, 0.5, 0.5], travel_lim=False):
        """
        Args
            n = side of 1D neighborhood, default = 70
            props = proportion of agents for [empty, agent1, agent2] default = [0.2, 0.4, 0.4]
            k = side of 1D neighborhood. default = 4
            thresholds = minimum proportion of like-agents in the neighborhood required by the agent. default = [0, 0.5, 0.5]
            travel_lim = restrictions on the mobility of the agent in the neighborhood, if True then agent can only move inside its 1D Moore-neighborhood. Default = False
        """
        self.n = n
        self.agents = [0, 1, 2] # we fix this so we only have [empty, agent1, agent2]
        self.props = props
        self.k = k
        self.thresholds = thresholds
        self.travel_lim = travel_lim
        
        # define our kernel which we use as the "radius from the agent" to get the nearby neighbors
        self.kernel = np.ones(shape=(self.k*2+1)) # it's k to the left and right of the agent
        self.kernel[(self.k*2)//2]=0 # we make the center 0 because we don't want to include the agent 
        
        # for stats tracking
        self.avg_similarities_1 = []
        self.avg_similarities_2 = []
        self.avg_similarities = []   
        self.run_num = 0 # run number
        
    def init_neighborhood(self, seed=None):
        """
        initialize neighborhood randomly given random seed (default=None) and class params
        """
        self.seed = seed
        if seed != None:
            np.random.seed(seed) # we can adjust the seed when we're running multiple samples so we still retain replicability
        else:
            pass
        
        # we randomly make a neighborhood based on proportions of agents and size n x n
        self.neighborhood = np.random.choice(self.agents, size=self.n, replace=True, p=self.props) 

    def run_one_episode(self):
        """
        Run one episode of the simulation (one agent moves)
        """
        # mode='same' just means that we return a matrix (of counts of similar neighbors surrounding each cell) with the same shape as the neighborhood
        # the conditions are just there to tell the function whether to count or not
        agent_1 = self.neighborhood == 1 # say we want to know the number of agents similar to 1
        agent_2 = self.neighborhood == 2 # say we want to know the number of agents similar to 2
        num_neighbors_1 = convolve(agent_1, self.kernel, mode='same')
        num_neighbors_2 = convolve(agent_2, self.kernel, mode='same')
        num_neighbors_total = num_neighbors_1 + num_neighbors_2

        ## PART 2
        empty = self.neighborhood == 0

        # now we find which agents are happy and which agents are not
        happy_1a = num_neighbors_1/num_neighbors_total
        happy_2a = num_neighbors_2/num_neighbors_total
        # for both agents
        frac_same = np.where(agent_1, happy_1a, happy_2a)

        # we don't want to confuse False as both empty and unhappy
        # for all arrays, let's indicate empty cells by np.nan 
        # let's also indicate for agents 1 and 2 only those pertaining to 1 or 2
        happy_1a = np.where(self.neighborhood==1, np.where(empty, np.nan, happy_1a), np.nan)
        happy_2a = np.where(self.neighborhood==2, np.where(empty, np.nan, happy_2a), np.nan)
        frac_same = np.where(empty, np.nan, frac_same)

        # get the proportion of same neighbors in locality of each agent
        avg_similarity_1 = np.nanmean(happy_1a)
        avg_similarity_2 = np.nanmean(happy_2a)
        avg_similarity = np.nanmean(frac_same) # proportion of same neighbots

        self.avg_similarities_1.append(avg_similarity_1)
        self.avg_similarities_2.append(avg_similarity_2)
        self.avg_similarities.append(avg_similarity)  

        # how many happy
        threshold_1 = self.thresholds[1] # threshold for agent 1
        happy_1 = happy_1a > threshold_1
        threshold_2 = self.thresholds[2] # threshold for agent 2
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
            current_agent = self.neighborhood[source] # note that the agent taggings must = index [0, 1, 2] otherwise pipe will fail
            agent_ = self.neighborhood == current_agent # we get similar neighbors
            num_neighbors_ = convolve(agent_, self.kernel, mode='same') # get the number of neighbors per cell
            threshold_ = self.thresholds[current_agent] # we use current agent as index of threshold
            happy_ = num_neighbors_/num_neighbors_total > threshold_

            # only get those with empty
            empty = self.neighborhood == 0
            happy_ = np.where(empty, happy_, np.nan) # if empty, we get it

            # with travel limits
            if self.travel_lim:
                dests = np.argwhere(happy_==True).flatten() # indices of possible destinations
                dests = [i for i in dests if i <= source+self.k and i >= source-self.k]# get destinations that are k away from source
                try: # if we can find a destination 
                    dest = np.random.choice(dests, size=1)[0]
                except: # randomly assign within travel limit
                    empty_inds = np.argwhere(empty==True).flatten() # indices of empty spots                    
                    empty_inds = [i for i in empty_inds if i <= source+self.k and i >= source-self.k]# get empty points that are k away from source                                    
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
                        pass # since we don't really change things

            # we now move the agent from the destination to the source
            # we then do this for several iterations
            self.neighborhood[dest] = current_agent
            self.neighborhood[source] = 0 # empty now   
        except:
            pass
        
        self.run_num += 1 # update run number        
        
    def init_image_folder(self):
        """
        initialize image folder
        """
        self.new_dir = f"charts/segregation_1d/seed_{self.seed}_n_{self.n}_k_{self.k}_props_{'_'.join([str(i) for i in self.props])}_thresh{'_'.join([str(i) for i in self.thresholds])}_travellim_{self.travel_lim}"             
        try:        
            os.mkdir(self.new_dir)        
        except:
            shutil.rmtree(self.new_dir)      
            os.mkdir(self.new_dir)        
        
    def save_neighborhood_plot(self):
        """
        We simply save the neighborhood plot for the run so we can load as gif later on
        """              

        # Plotting
        f = plt.figure(figsize=(15, 2))
        plt.imshow(self.neighborhood.reshape(-1, 1).T)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{self.new_dir}/{str(self.run_num).zfill(3)}.png')
        plt.close()
        
    def plot_similarity_1d(self):
        """
        Plot similarities over time for agent 1, 2, and all agent average
        """

        print('final avg similarities agent 1, agent 2, all ', self.avg_similarities_1[-1], self.avg_similarities_2[-1], self.avg_similarities[-1])
        plt.plot(self.avg_similarities_1, label='1')
        plt.plot(self.avg_similarities_2, label='2')
        plt.plot(self.avg_similarities, label='all')
        plt.xlabel('time')
        plt.ylabel('similarity between neighborhoods')
        plt.legend();
        
    def make_gif(self):
        """
        We make gif animation of the simulation
        """
        # save gif
        filenames = glob.glob(f"{self.new_dir}/*.png")
        filenames = sorted(filenames)
        filenames = filenames + [filenames[-1]] * 15 # so we freeze at the output
        images=[]
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{self.new_dir}/movie.gif', images)    
        plt.show()

    def show_gif(self):
        with open(f'{self.new_dir}/movie.gif','rb') as f:
            display(Image(data=f.read(), format='png'))