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
from scipy.stats import invgauss

def plot_thresh_gif(dir_):
    with open(f'{dir_}/movie.gif','rb') as f:
        display(Image(data=f.read(), format='png'))
        

# since we now know the fraction of like-agents at any one time,
# we can define a threshold distribution per agent and that will define whether they will activate or not
# define a distribution
def linear_dist(x):
    # accepts an array and applies a function
    # must be between 0 and 1
    y = x/np.max(x)
    return y

# thresholds from a normal distribution - this is what Granovetter used
# returns proportion of people having the given threshold
def norm_dist(x, mean, stdev):
    np.random.seed(42)
    y = np.random.normal(loc=mean, scale=stdev, size=len(x))
    y = np.array(sorted(y))
    y = np.clip(y, 0, 1)
    return y

# inverse gaussian
def inv_norm_dist(x, mean):
    np.random.seed(42)
    y = invgauss.rvs(mean, size=len(x))
    y = np.array(sorted(y))
    y = np.clip(y, 0, 1)
    return y    

def exp_dist(x, a, b, c):
    np.random.seed(42)
    y = a ** (x + b) + c
    y = np.array(sorted(y))
    y = np.clip(y, 0, 1)
    return y    

# proportion activated tracker
def run_thresh_sim(neighborhood, props, thresh, new_dir):
    
    try:
        os.mkdir('charts/threshold/')
    except:
        pass
    
    try:
        os.mkdir(new_dir)
        print(f'{new_dir} created')
    except:
        print(f'{new_dir} exists')    

    rs = []
    for iter_ in range(30):

        r1 = neighborhood.sum() / len(props) # proportion activated
        r2 = sum(thresh <= r1)/len(thresh)

        rs.append(r1)

        to_activate = int(len(props) * r2) - neighborhood.sum() # increemental agents to activate
        if to_activate > 0: # we will activate more
        # randomly select m zeros to activate
            zero_coords = [(i[0][0], i[0][1]) for i in zip(np.argwhere(neighborhood==0))]

            selected_inds = np.random.choice(range(len(zero_coords)), size=to_activate, replace=False)
            selected_coords = np.array(zero_coords)[selected_inds]    

            for i in range(len(selected_coords)):
                neighborhood[(selected_coords[i][0], selected_coords[i][1])] = 1        

        if to_activate < 0: # we will deactivate
        # randomly select m zeros to activate
            one_coords = [(i[0][0], i[0][1]) for i in zip(np.argwhere(neighborhood==1))]

            selected_inds = np.random.choice(range(len(one_coords)), size=np.abs(to_activate), replace=False)
            selected_coords = np.array(one_coords)[selected_inds]    

            for i in range(len(selected_coords)):
                neighborhood[(selected_coords[i][0], selected_coords[i][1])] = 0           

        f = plt.figure(figsize=(4, 4))
        plt.imshow(neighborhood)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{new_dir}/{str(iter_).zfill(5)}.png')
        plt.close()    

    # save gif
    filenames = glob.glob(f"{new_dir}/*.png")
    filenames = sorted(filenames)
    filenames = filenames + [filenames[-1]] * 20 # so we freeze at the output
    images=[]
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{new_dir}/movie.gif', images)
    
    return rs
