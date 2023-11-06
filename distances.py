import math
import numpy as np  
def calculate_distance(gt_coordonates , pred_coordonates):
    
    
    if len(pred_coordonates) > 0:
        # print(gt_coordonates,pred_coordonates)
        

        dx = gt_coordonates[0]-pred_coordonates[0]
        dy = gt_coordonates[1]-pred_coordonates[1]

        print(dx, dy)
        distance = math.sqrt(dx*dx+dy*dy)

        print(distance)
   
    return distance


def pixels2mm(coordonate_pixel, magnification_factor, image_spacing):

 
    img_space = np.array([image_spacing[0], image_spacing[1]])
    mm = np.array([coordonate_pixel[0],coordonate_pixel[1]]) * img_space / magnification_factor
    # print("Imgspace",image_spacing,'magfactor',magnification_factor)
    # print('mm',mm)
    

    return mm



def mm2pixels(coordonate_mm_list, magnification_factor, image_spacing):

  
    
    pixeli = np.array(coordonate_mm_list[0], coordonate_mm_list[1])/np.array(image_spacing) * magnification_factor
      

    return pixeli