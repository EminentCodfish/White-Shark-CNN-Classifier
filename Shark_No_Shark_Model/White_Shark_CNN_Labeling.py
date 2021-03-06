# -*- coding: utf-8 -*-
"""
White Shark CNN labeling script

Last updated: 10/17/2019
Author: Chris Rillahan
"""

#Import packages
import os, cv2, logging
from fastai.vision import *
from iptcinfo3 import IPTCInfo
from tqdm import trange

#Starting parameters
path = "D:\\Mark recapture videos 2015\\"

#Load the pre-trained CNN 
defaults.device = torch.device('cuda')
cnn_path = 'D:\\Population Study Videos\\Training Data\\'
learn = load_learner(cnn_path, 'WS_RESNET_50.pkl')

#Filewalker function
for root, dirs, filenames in os.walk(path):
    for f in filenames:
        if f[-4:] == '.MP4':
            print('Starting to label: ' + str(root) + '\\' + str(f))
            video_name = str(root) + '\\' + str(f)
            
            #Set-up the export path and export folder
            export_path = root + '\\' + f[:-4]
            if not os.path.exists(export_path): 
                os.makedirs(export_path)
                os.makedirs(export_path + '\\Shark\\')
                os.makedirs(export_path + '\\No_Shark\\')
                
            #Open the GoPro video
            video = cv2.VideoCapture(video_name)
            
            #Get the videos specs.
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            FPS = video.get(cv2.CAP_PROP_FPS)
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            size = (int(width), int(height))
            #print(total_frames, FPS, size)
            
            #Set-up the exported labeled video
            current_frame = 0
            status = video.isOpened()
            if status == True:
                codec = cv2.VideoWriter_fourcc(*'h264')
                video_out = cv2.VideoWriter(video_name[:-4] + '_labeled.mp4', 
                                            codec, FPS, size, 1)
                
                #Loop through the video frames
                while current_frame < (total_frames-1):
                    for i in trange(int(total_frames-1)):
                        success, image = video.read()
                        if success == True:
                            #cv2.imwrite(path + 'frame.jpg', image)
                            #img = open_image(path + 'frame.jpg')
                            t = torch.tensor(np.ascontiguousarray(np.flip(image, 2)).transpose(2,0,1)).float()/255
                            img = Image(t) # fastai.vision.Image, not PIL.Image
                            
                            #Run the image through the CNN classifier
                            pred_class,pred_idx,outputs = learn.predict(img)
                            #print(pred_class, outputs[pred_idx]) #Look at the outputs
                            
                            #Add the label and probability to the image
                            text_out = str(pred_class) + ': ' + str(outputs[pred_idx])
                            cv2.putText(img = image, text = text_out, org = (50,50), fontFace = cv2.FONT_HERSHEY_PLAIN,
                                        fontScale = 2, color = (255,255,255))
                            
                            #Save labeled image
                            cv2.imwrite(path + 'frame.jpg', image)
                            
                            #Add metadata
                            if str(pred_class) == 'No shark':                               
                                #Label and save the image
                                logging.basicConfig(level=logging.CRITICAL)#Turn off output; it's annoying
                                info = IPTCInfo(path + 'frame.jpg', force = True)
                                info['keywords'] = ['No shark']
                                info.save_as(export_path + '\\No_Shark\\' + f[:-4] + '_' + str(int(current_frame)) + '.jpg')
                                logging.basicConfig(level=logging.INFO)#Turn the output back on.
                                
                            if str(pred_class) == 'Shark':
                                #Label and save the image
                                logging.basicConfig(level=logging.CRITICAL)
                                info = IPTCInfo(path + 'frame.jpg', force = True)
                                info['keywords'] = ['Shark']
                                info.save_as(export_path + '\\Shark\\' + f[:-4] + '_' + str(int(current_frame)) + '.jpg')
                                logging.basicConfig(level=logging.INFO)
                            
                            #Write the image to the labeled video file
                            video_out.write(image)
                            
                            #Show the image
                            #cv2.imshow('Video', image)
                            
                            #Update the current
                            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)

                            k = cv2.waitKey(10)
                            if k == 27:
                                current_frame = total_frames    
                        pass
    
                video.release()
                video_out.release()
                cv2.destroyAllWindows()
            else:
                print("Error: Video could not be loaded.")