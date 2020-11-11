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

# thresholds from a normal distribution - this is what Granovetter used
# returns proportion of people having the given threshold
def norm_dist(x, mean, stdev):
    np.random.seed(42)
    y = np.random.normal(loc=mean, scale=stdev, size=len(x))
    y = np.array(sorted(y))
    y = np.clip(y, 0, 1)
    return y

def plot_standing_ovation(**params):
    n = params['n']
    q_thresh = params['q_thresh']   
    seed=params['seed']

    dir_ = f"charts/standing_ovation/seed_{seed}_n_{n}_q_mean_{q_mean}_q_stdev_{q_stdev}_p_mean_{p_mean}_p_stdev_{p_stdev}"   
    with open(f'{dir_}/movie.gif','rb') as f:
        display(Image(data=f.read(), format='png'))
        
# putting it all together
def standing_ovation(**params):
    
    n = params['n']    
    q_thresh = params['q_thresh']     
    q_mean = params['q_mean']    
    q_stdev = params['q_stdev']    
    p_mean = params['p_mean']      
    p_stdev = params['p_stdev']        
    kernel = params['kernel']    
    num_iters=params['num_iters']
    save_step=params['save_step']     
    seed = params['seed']   
    
    try:
        os.mkdir('charts/standing_ovation/')
    except:
        pass
    
    new_dir = f"charts/standing_ovation/seed_{seed}_n_{n}_q_mean_{q_mean}_q_stdev_{q_stdev}_p_mean_{p_mean}_p_stdev_{p_stdev}"   
    try:        
        os.mkdir(new_dir)        
    except:
        shutil.rmtree(new_dir)      
        os.mkdir(new_dir)          
                                                                                                                         
    # Part 1
    # initialize matrices
    np.random.seed(seed)
    act_matrix = np.zeros(shape=(n, n)) # whether standing or not
    
    # generate thresholds
    props = np.arange(0, 1, 1/(n**2)) # agents are indices, value is the (cumulative proportion)
    p_thresh = norm_dist(props, mean=p_mean, stdev=p_stdev) # agents are indices thresholds for peer pressure    
    np.random.shuffle(p_thresh)
    
    q_matrix = norm_dist(props, mean=q_mean, stdev=q_stdev) # get a quality assessment matrix from a (normal) distribution
    np.random.shuffle(q_matrix)
    q_matrix = q_matrix.reshape((n, n))

    q_thresh_matrix = np.ones(shape=(n, n)) * q_thresh # quality thresholds e.g. 0.5
    p_thresh_matrix = np.zeros(shape=(n, n)) # indicating agent's threshold re proportion of others standing up

    # # we get the coordinates
    coords = [(i[0][0], i[0][1]) for i in zip(np.argwhere(p_thresh_matrix==0))]

    # we assign agents indicated by their thresholds randomly into the matrix
    for i in range(len(coords)):
        x, y = coords[i][0], coords[i][1]
        p_thresh_matrix[x, y] = p_thresh[i]

    # those whose quality thresholds are breached will activate
    act_matrix_init = (q_matrix > q_thresh_matrix).astype(int) # initialize, we need this as the permanent standing agents
    act_matrix = act_matrix_init.copy() # this is what we update per time  
    
    # PART 2
    
    props_act = []
    for iter_ in range(num_iters):
        # prop activated
        prop_act = act_matrix.sum() / (n * n)
        props_act.append(prop_act)
        
        # number of activated agents in field of vision
        options = dict(mode='same', boundary='fill') # here's a neat way to define arguments
        act_perceived = correlate2d(act_matrix, kernel, **options) # perceived activation
        all_perceived = correlate2d(np.ones(shape=(n,n)), kernel, **options) # all agents in field of view
        act_prop_perceived = act_perceived / all_perceived

        # those who will activate at t + 1
        act_new = (act_prop_perceived > p_thresh_matrix).astype(int)

        # those who will deactivate at t + 1
        deact_new = (act_prop_perceived < p_thresh_matrix).astype(int)

        # update activation matrix
        act_matrix = ((act_matrix + act_new) != 0).astype(int)

        # those who deactivate will sit down while those who stood up from quality will remain standing up
        # reverse deact new so we can do elementwise multiplication to deactivate
        deact_new_inv = 1 - deact_new
        act_matrix = np.multiply(act_matrix, deact_new_inv) + act_matrix_init # we deactivate but add back permanently standing agents
        act_matrix = (act_matrix != 0).astype(int)

        # Plotting
        # save only by 10 iterations (so it's faster)
        if iter_%save_step == 0:
            f = plt.figure(figsize=(4, 4))
            plt.imshow(act_matrix)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'{new_dir}/{str(iter_).zfill(5)}.png')
            plt.close()
        
    print('sim complete')        

    # save gif
    filenames = glob.glob(f"{new_dir}/*.png")
    filenames = sorted(filenames)
    filenames = filenames + [filenames[-1]] * 20 # so we freeze at the output
    images=[]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{new_dir}/movie.gif', images)
    plt.show()
    return props_act
        