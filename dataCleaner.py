'''
Original data is in the form data/train, data/test. We want to label them inside of the file name, and finally once they are
all together split them into train, test & validate
'''

# #imports
# import os

# os.chdir('./CancerDataset/data/malignant')

# for filename in os.listdir('.'):
#     length = len(filename)-2
#     length = length*-1
#     # print(filename[length:])
#     #from that length, we want to only get length-2 last bit of the string
#     os.rename(filename, '1_'+filename[length:])

# import subprocess

# ## define your paths
# path1 = './CancerDataset/data/malignant/'
# path2 = './CancerDataset/data/malignant_2/'

# ## where to place the merged data
# merged_path = './CancerDataset/data/malignant_final/'

# ## write an rsync commands to merge the directories
# rsync_cmd = 'rsync' + ' -avzh ' + path1 + ' ' + path2 + ' ' + merged_path

# ## run the rsync command
# subprocess.run(rsync_cmd, shell=True)

import splitfolders

input = 'CancerDataset/data/input'
output = 'CancerDataset/output'

splitfolders.ratio(input, output=output, seed=1337, ratio=(0.8,0.1,0.1))