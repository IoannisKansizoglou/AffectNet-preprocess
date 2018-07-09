
########################### LIBRARIES ##########################

import os, os.path, time
import pandas as pd
import numpy as np

########################## PARAMETERS ##########################

# Starting time
t0 = time.time()
# Path of dataset
datasetPATH = '/media/gryphonlab/8847-9F6F/Ioannis/AffectNet/'
# Path of excel files for training and evaluation data
targetPATH = '/home/gryphonlab/Ioannis/Works/AffectNet/'


########################## FUNCTIONS ###########################


### Function for creating folders ###
def create_folder(PATH):
    
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print ('Error: Creating directory' + PATH)


### Function to convert emotion number to class corresponding number ###
def convert_emotion(emotion):

    if emotion == 1:
        label = 0
    elif emotion == 2:
        label = 2
    elif emotion == 3:
        label = 1
    elif emotion == 4:
        label = 3
    elif emotion == 5:
        label = 5
    elif emotion == 6:
        label = 4
    else:
        label = -1

    return label


### Function to choose AffectNet's images that have suitable labels ###
def choose_data(images, emotions):

    faces, labels = list(), list()

    for i in range(np.shape(emotions)[0]):
        
        label = convert_emotion(emotions[i])

        if label >= 0:

            labels.append(label)
            faces.append(datasetPATH+images[i])
        
    faces = np.array(faces)    
    labels = np.array(labels)

    return faces, labels


############################# MAIN ############################


def main():

    train_data = pd.read_csv(targetPATH+'training.csv')
    train_images = np.array(train_data['subDirectory_filePath'])
    train_emotions = np.array(train_data['expression'])
    train_faces, train_labels = choose_data(train_images, train_emotions)

    eval_data = pd.read_csv(targetPATH+'validation.csv')
    eval_images = np.array(eval_data['subDirectory_filePath'])
    eval_emotions = np.array(eval_data['expression'])
    eval_faces, eval_labels = choose_data(eval_images, eval_emotions)

    d1 = {'train_faces': train_faces, 'train_labels': train_labels}
    d2 = {'eval_faces': eval_faces, 'eval_labels': eval_labels}

    create_folder(targetPATH+'Core')

    df1 = pd.DataFrame(data=d1)
    df2 = pd.DataFrame(data=d2)
    #df3 = pd.DataFrame(data=d3)
    df1.to_csv( targetPATH + 'Core/training_data.csv' )
    df2.to_csv( targetPATH + 'Core/evaluation_data.csv' )

    # Execution time
    print( 'Execution time of createInputcsv.py [sec]: ' + str(time.time() - t0) )


# Control runtime
if __name__ == '__main__':
    main()