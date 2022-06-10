import cv2
import os
import shutil

VIDEO_PATH = "./video1.mp4"

def extract_images_from_video(video_path):
    video = cv2.VideoCapture(VIDEO_PATH)

    success = True
    framenumber = 0
    img_number = 0
    path = "./frames/img0"

    while success:
        if framenumber%20==0:
            success,frame = video.read()
            if success:
                cv2.imwrite((path+str(img_number)+".jpg"),frame)
            img_number+=1
        framenumber+=1

def divide_batches_into_folders(images_per_batch):
    dest_path_base = "/images/batch"
    folder_counter = 0
    for imagenumber in range(len(os.listdir("./frames"))):
        image = "img0"+str(imagenumber)+".jpg"
        if imagenumber%images_per_batch == 0:
            os.chdir("./images")
            dest_path = dest_path_base+str(folder_counter)
            os.mkdir("batch"+str(folder_counter))
            os.chdir("../")
            folder_counter+=1
            
        shutil.move(str(os.getcwd())+"/frames/"+image, str(os.getcwd())+dest_path+"/")
        


#extract_images_from_video(VIDEO_PATH)
divide_batches_into_folders(50)
        
    