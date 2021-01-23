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

# putting them all together in a function
class segregate_2d:
    def __init__(self, n = 20, props = [0.2, 0.4, 0.4], kernels = [3, 3], 
                 thresholds = [0.5, 0.5], thresholds_max = [1, 1], travel_lim=False):
        """
        Args
            n = side of 2D neighborhood, default = 20
            props = proportion of agents for [empty, agent1, agent2] default = [0.2, 0.4, 0.4]
            kernels = sides of 2D Moore neighborhood per agent. default = [3, 3]
            thresholds = minimum proportion of like-agents in the neighborhood required by the agent excluding empty cells. default = [0.5, 0.5]
            thresholds_max = maximum proportion of like-agents in the neighborhood required by the agent e.g. default = [1, 1]
            travel_lim = restrictions on the mobility of the agent in the neighborhood, if True then agent can only move inside its 1D Moore-neighborhood. Default = False
        """
        self.n = n
        self.agents = [0, 1, 2] # we fix this so we only have [empty, agent1, agent2]
        self.props = props
        self.thresholds = thresholds
        self.thresholds_max = thresholds_max
        self.travel_lim = travel_lim
        self.kernels = kernels
        self.seed = None # initialize
        
        # make kernel per agent (can be different)
        k1 = kernels[0]
        k2 = kernels[1]    
        self.kernel_1 = np.ones(shape=(k1, k1))
        self.kernel_1[k1//2, k1//2] = 0 # we make the center of the kernel 0 because we don't want to count the agent in the center
        self.kernel_2 = np.ones(shape=(k2, k2))
        self.kernel_2[k2//2, k2//2] = 0
        
        # for stats tracking
        self.avg_similarities_1 = []
        self.avg_similarities_2 = []
        self.avg_similarities_all = []   
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
    
        ## PART 1
        # here we break down, the algo to count neighbors per agent
        # we randomly make a neighborhood based on proportions of agents and size n x n
        self.neighborhood = np.random.choice(self.agents, size=(self.n, self.n), replace=True, p=self.props) 
    
    def run_one_episode(self):
        """
        Run one episode of the simulation (one agent moves)
        """

        # counting the number of neighbors using correlate2d
        # in Think Complexity, they used boundary="wrap"
        # this "tiles" the neighborhood matrix so for the cell in the top right corner, we consider the cells in the left and bottom areas
        # Here, we don't do that so we use boudary="fill" or we pad with nulls. This implies that the neighborhood has edges like in real life.
        # Note that Schelling made this same assumption that there are edges.
        # mode (counts) surrounding a cell can thus be calculated by running the "neighborhood window" or kernel from top left to bottom right.
        # mode='same' just means that we return a matrix (of counts of similar neighbors surrounding each cell) with the same shape as the neighborhood
        # the conditions are just there to tell the function whether to count or not
        agent_1 = self.neighborhood == 1 # say we want to know the number of agents similar to 1
        agent_2 = self.neighborhood == 2 # say we want to know the number of agents similar to 1

        options = dict(mode='same', boundary='fill') # here's a neat way to define arguments
        num_neighbors_1 = correlate2d(agent_1, self.kernel_1, **options)
        num_neighbors_2 = correlate2d(agent_2, self.kernel_2, **options)
        num_neighbors_all = num_neighbors_1 + num_neighbors_2 # so we exclude empty cells

        ## PART 2
        # now we find which agents are happy and which agents are not
        # we use the threshold proportions
        frac_neighbors_1 = num_neighbors_1 / num_neighbors_all
        frac_neighbors_2 = num_neighbors_2 / num_neighbors_all
        frac_neighbors_all = np.where(agent_1, frac_neighbors_1, frac_neighbors_2) # if agent_1, use agent 1 props, else agent 2
        
        # remember to just select the agent (1, or 2) and remove empty cells
        is_empty = self.neighborhood == 0    
        frac_neighbors_all = np.where(is_empty, np.nan, frac_neighbors_all) # then we remove empty       
        
        # making dissimilar/empty cells around an agent null
        frac_neighbors_1 = np.where(self.neighborhood==1, frac_neighbors_1, np.nan)
        frac_neighbors_2 = np.where(self.neighborhood==2, frac_neighbors_2, np.nan)       

        # getting happy agents        
        threshold_1 = self.thresholds[0] # threshold of agent 1
        threshold_2 = self.thresholds[1] # threshold of agent 2         
        thresholds_max_1 = self.thresholds_max[0] # max thresh for agent 1
        thresholds_max_2 = self.thresholds_max[1] # max thresh for agent 2
        
        frac_happy_1 = (frac_neighbors_1 >= threshold_1) & (frac_neighbors_1 <= thresholds_max_1)
        frac_happy_2 = (frac_neighbors_2 >= threshold_2) & (frac_neighbors_2 <= thresholds_max_2)  
        
        # let's be sure and remove dissimilar agents
        frac_happy_1 = np.where(self.neighborhood==1, frac_happy_1, np.nan)
        frac_happy_2 = np.where(self.neighborhood==2, frac_happy_2, np.nan)        
        
        # remove empty cells    
        frac_happy_all = np.where(agent_1, frac_happy_1, frac_happy_2) # if agent_1, use agent 1 happiness, else agent 2
        frac_happy_all = np.where(is_empty, np.nan, frac_happy_all) # then we remove empty

        # calculate the proportion of like-neighbors per agent
        avg_frac_1 = np.nanmean(frac_neighbors_1)
        avg_frac_2 = np.nanmean(frac_neighbors_2)
        avg_frac_all = np.nanmean(frac_neighbors_all) 
        
        # for tracking of segregation over time
        self.avg_similarities_1.append(avg_frac_1)
        self.avg_similarities_2.append(avg_frac_2)
        self.avg_similarities_all.append(avg_frac_all)        

        # PART 3
        # get the coordinates of unhappy cells
        unhappy_coords = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(frac_happy_all==0)))] # this is two lists that we need to convert to a (x, y)

        # if there are no more unhappy coordinates
        if len(unhappy_coords)==0:
            pass
        
        else:
            # get the coordinates of empty cells
            empty_coords = [(i[0][0], i[0][1]) for i in list(zip(np.argwhere(np.isnan(frac_happy_all))))]

            # switch places one agent at a time        
            unhappy_agent_coord_ind = np.random.choice(range(len(unhappy_coords)))
            unhappy_agent_coord = unhappy_coords[unhappy_agent_coord_ind]
            unhappy_agent = self.neighborhood[unhappy_agent_coord]

            if self.travel_lim:
                if unhappy_agent == 1:
                    k = self.kernels[0] # get the kernel size for agent 1
                elif unhappy_agent == 2:
                    k = self.kernels[1] # get the kernel size for agent 2
            # we limit the possible empty locations to only that surrounding the selected agent
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
                    pass # just go to the next iteration      

            else: # no travel limits    
                empty_coord_ind = np.random.choice(range(len(empty_coords)))
                empty_coord = empty_coords[empty_coord_ind]      

            # then we switch an unhappy cell with an empty cell
            self.neighborhood[empty_coord] = unhappy_agent
            self.neighborhood[unhappy_agent_coord] = 0
        self.run_num += 1
        

    def init_image_folder(self):
        """
        initialize image folder
        """
        self.new_dir = f"charts/segregation_2d/seed_{self.seed}_n_{self.n}_kernels_{'_'.join([str(i) for i in self.kernels])}_props_{'_'.join([str(i) for i in self.props])}_thresh{'_'.join([str(i) for i in self.thresholds])}_travellim_{self.travel_lim}"
        try:        
            os.mkdir(self.new_dir)        
        except:
            shutil.rmtree(self.new_dir)      
            os.mkdir(self.new_dir)   
            
    def save_neighborhood_plot(self, save_step = 1):
        """
        We simply save the neighborhood plot for the run so we can load as gif later on        
        Arg: save_step = at what step size do we save (so we don't save each step), default = 1
        """               
        # Plotting
        # save only by 10 iterations (so it's faster)
        if self.run_num%save_step == 0:
            f = plt.figure(figsize=(4, 4))
            plt.imshow(self.neighborhood)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'{self.new_dir}/{str(self.run_num).zfill(5)}.png')
            plt.close()
        
    def plot_similarity_1d(self):
        """
        Plot similarities over time for agent 1, 2, and all agent average
        """
        print('final avg similarities agent 1, agent 2, all ', self.avg_similarities_1[-1], self.avg_similarities_2[-1], self.avg_similarities_all[-1])
        plt.plot(self.avg_similarities_1, label='1')
        plt.plot(self.avg_similarities_2, label='2')
        plt.plot(self.avg_similarities_all, label='all')
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