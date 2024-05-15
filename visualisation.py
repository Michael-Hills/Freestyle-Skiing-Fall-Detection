import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pyts.image import GramianAngularField
from PIL import Image



def cosineGraph():

    """Cosine similarity graph"""

    columns = ['Pelvis Accel Sensor X,mG', 'Pelvis Accel Sensor Y,mG',
       'Pelvis Accel Sensor Z,mG', 'Pelvis Rot X,', 'Pelvis Rot Y,',    
       'Pelvis Rot Z,']

    #Get file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallFilePath = os.path.join(script_dir, "3S/RawFall")
    fallFiles = glob.glob(os.path.join(fallFilePath , "*.csv")) 

    df = pd.read_csv(fallFiles[0],usecols=columns,header=0,dtype={'MarkerNames': str})
    x = df.values.T


    plt.figure(figsize=(12, 5))
    for i in range(6):
        plt.plot(x[i], label=f'Channel {i + 1}')
    plt.legend()
    plt.title('Norm')

    plt.show()

    x_cosine = cosine_similarity(x.T)

    # Plot the corresponding image
    plt.figure(figsize=(8, 8))
    plt.imshow(x_cosine, cmap='rainbow', origin='lower', vmin=-1, vmax=1)
    plt.show()
    


def aggregatedGramian():

    """Concatenated GAF"""

    columns = ['Pelvis Accel Sensor X,mG', 'Pelvis Accel Sensor Y,mG',
       'Pelvis Accel Sensor Z,mG', 'Pelvis Rot X,', 'Pelvis Rot Y,',    
       'Pelvis Rot Z,']

    #Get file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallFilePath = os.path.join(script_dir, "3S/RawNonFall")
    fallFiles = glob.glob(os.path.join(fallFilePath , "*.csv")) 


    # create image for each sample
    for i in range(len(fallFiles)):

        df = pd.read_csv(fallFiles[i],usecols=columns,header=0,dtype={'MarkerNames': str})

        name =  fallFiles[i].split("\\")[-1]

        # calculate the gramian angular field
        x = df.values.T
        gaf = GramianAngularField(method='summation')
        x_gaf = gaf.fit_transform(x)

        count = 1

        script_dir = os.path.dirname(os.path.abspath(__file__))
        tempFilePath = os.path.join(script_dir, "tempImages")
        filePath =  os.path.join(script_dir, "Images/nonFall")
    

        # create and save the image for each GAF
        for x in x_gaf:
            filename = tempFilePath + "/Figure" + str(count) + ".png"
            plt.imshow(x, cmap='rainbow', origin='lower', vmin=-1, vmax=1)
            plt.axis('off')
            im = plt.savefig(filename,pad_inches=0,bbox_inches="tight")
            plt.close()

            count += 1

        images = [Image.open(x) for x in [tempFilePath+'/Figure1.png', tempFilePath+'/Figure2.png',tempFilePath+'/Figure3.png',
                                        tempFilePath+'/Figure4.png',tempFilePath+'/Figure5.png',tempFilePath+'/Figure6.png']]
        


        # resize so the 6 images make a square
        new_images = []
        widths, heights = zip(*(i.size for i in images))
        for image in images:
            new_image = image.resize((int(widths[0]*1.5), heights[0]))
            new_images.append(new_image)


        widths, heights = zip(*(i.size for i in new_images))
        
        total_width = max(widths) * 2
        max_height = max(heights) * 3

        new_im = Image.new('RGB', (total_width, max_height))


        # concatenate the images together
        x_offset = 0
        for im in new_images[0:2]:
            new_im.paste(im, (x_offset,0))
            x_offset += widths[0]

        x_offset = 0
        for im in new_images[2:4]:
            new_im.paste(im, (x_offset,heights[0]))
            x_offset += widths[0]

        x_offset = 0
        for im in new_images[4:6]:
            new_im.paste(im, (x_offset,heights[0]*2))
            x_offset += widths[0]

        # save the concatenated image to its respective folder
        new_im.save(filePath+'/'+name[:-4]+'.png')

        
    return


