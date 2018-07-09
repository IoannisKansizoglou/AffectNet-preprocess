
########################### LIBRARIES ##########################

import os, os.path, time
import pandas as pd
import numpy as np
import subprocess

########################## PARAMETERS ##########################

# Starting time
t0 = time.time()
# Path of dataset
datasetPATH = '/media/gryphonlab/8847-9F6F/Ioannis/AffectNet/'
# Path of excel files for training and evaluation data
targetPATH = '/home/gryphonlab/Ioannis/Works/AffectNet/Core/'


########################## FUNCTIONS ###########################


### Function for creating folders ###
def create_folder(PATH):
    
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print ('Error: Creating directory' + PATH)



############################# MAIN ############################


def main():

    data = pd.read_csv(targetPATH+'training_data.csv')
    paths = data['train_faces']
    '''
    for path in paths:
        
        if path[-3:] == 'BMP':

            print(path)
            oldpath = path
            path = path[:-3]+'jpg'
            command = 'sudo convert ' + oldpath + ' ' + path
            subprocess.call(command, shell=True)
        
    d = {'eval_faces': paths, 'eval_labels': data['eval_labels']}
    df = pd.DataFrame(data=d)
    df.to_csv( targetPATH + 'evaluation_data.csv' )
    '''

    for path in paths:

        if path[-3:] != 'png' and path[-3:] != 'peg' and path[-3:] != 'jpg' and path[-3:] != 'PNG' and path[-3:] != 'PEG' and path[-3:] != 'JPG':

            print(path)

    # Execution time
    print( 'Execution time of createInputcsv.py [sec]: ' + str(time.time() - t0) )


# Control runtime
if __name__ == '__main__':
    main()