#处理数据是视频,如何测试抽取好的图像
#不同模型要修改model_path，trainonDFDC
#不同测试数据集要修改sample，label
import sys, os
import cv2
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import torch
import torch.nn as nn
import glob
import torch.optim as optim
import numpy as np
from time import perf_counter

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score,auc
from torchvision import transforms
import pandas as pd
import json
import face_recognition
import random
from concurrent.futures import ThreadPoolExecutor

from config_trafuse import GlobalConfigs
from model.transfuse_effi0619 import RGBSRMfuser #backbone is efficientb0
from model.transfuse2class import ResRGBSRMfuser  #backbone is resnet
from model.transfuse import Res34Res18 #RES_RGBFAD

sys.path.insert(1,'helpers')
sys.path.insert(1,'model')
sys.path.insert(1,'weight')


from helpers.helpers_read_video_1 import VideoReader
from helpers.helpers_face_extract_1 import FaceExtractor

device = 'cuda' #if torch.cuda.is_available() else 'cpu'

from helpers.blazeface import BlazeFace
facedet = BlazeFace().to(device)
facedet.load_weights("helpers/blazeface.pth")
facedet.load_anchors("helpers/anchors.npy")
_ = facedet.train(False)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize_transform = transforms.Compose([
        transforms.Normalize(mean, std)]
)

tresh=50
sample='../../test-video/NeuralTextures/'
#../../celeb-synthesis-test  ../../celeb-real-test  ../../youtube-real-test
#sample__prediction_data/   ../CViT-main/preprocessing/dfdc_data/dfdc_train_part_49/
#../../NeuralTextures/c40/videos/
#../../test-video/Deepfakes/
#../../coderepo/dataset/FaceSwap/c23/videos
OUTPUT_DIR = 'result/'#保存结果图像
model_path = 'weight/ResnetRGBSRM_ep35Pretrainbestacc.pth'#
#能够预测的：
# transfuse_efficientnetb0_ep35layer1AdammlpPretrain.pth #eff:RGB+FDA
# RGBSRM_ep35layer1AdammlpPretrain.pth#eff:RGB SRM
# ResnetRGBSRM_ep35Pretrainbestacc.pth  #resnet:RGBSRM
trainonDFDC = True

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ran = random.randint(0,400)
ran_min = abs(ran-1)

filenames = sorted([x for x in os.listdir(sample) if x[-4:] == ".mp4"]) #[ran_min, ran] -  select video randomly
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)

config = GlobalConfigs()
config.n_layer = 1
#load model
model = ResRGBSRMfuser(config)
model.to(device)

checkpoint = torch.load(model_path) # for GPU
model.load_state_dict(checkpoint['state_dict']) #model.load_state_dict(checkpoint)
_ = model.eval()

def predict_on_video(dfdc_filenames, num_workers):
    def process_file(i):
        filename = dfdc_filenames[i]
        print(filename)
        decCViT = predict(os.path.join(sample, filename), tresh, mtcnn)
        return decCViT

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(dfdc_filenames)))
    return list(predictions)

# MTCNN face extctaction
def face_mtcnn(frame, face_tensor_mtcnn):#frame是一帧
    mtcnn_frame = mtcnn.detect(frame)
    temp_face = np.zeros((5, 224, 224, 3), dtype=np.uint8)#最多有5个人脸
    count=0
    if count<5 and (mtcnn_frame[0] is not None):
        for face_mt in mtcnn_frame[0]:
            x1, y1, width_, height_ = face_mt
            face_mt= frame[int(y1):int(height_), int(x1):int(width_)]
            if face_mt.size>0 and (count<5):
                resized_image_mtcnn = cv2.resize(face_mt, (224, 224), interpolation=cv2.INTER_AREA) 
                resized_image_mtcnn = cv2.cvtColor(resized_image_mtcnn, cv2.COLOR_RGB2BGR)
                temp_face[count]=resized_image_mtcnn
                count+=1
    if count == 0:
        return [],0
    return temp_face[:count], count

# face_recognition face extctaction
def face_face_rec(frame, face_tensor_face_rec):
    
    face_locations = face_recognition.face_locations(frame)
    temp_face = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    count=0
    for face_location in face_locations:
        if count<5:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            #face_image = Image.fromarray(face_image)
            temp_face[count]=face_image
            count+=1
    if count == 0:
        return [],0
    return temp_face[:count], count

# blazeface face extctaction
def face_blaze(video_path, filename, face_tensor_blaze):

    frames_per_video = 45  
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_random_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn, facedet)
    
    faces = face_extractor.process_video(video_path)
    # Only look at one face per frame.
    #face_extractor.keep_only_best_face(faces)
  
    count_blaze=0
    
    temp_blaze = np.zeros((45, 224, 224, 3), dtype=np.uint8)
    for frame_data in faces:
        for face in frame_data["faces"]:
                if count_blaze<44:
                    resized_facefrm = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_facefrm = cv2.cvtColor(resized_facefrm, cv2.COLOR_RGB2BGR)
                    temp_blaze[count_blaze]=resized_facefrm
                    count_blaze+=1
    if count_blaze==0:
        return [],0
    return temp_blaze, count_blaze

#y_pred=0
def predict(filename, tresh, mtcnn):#单个视频的检测
    store_faces=[]
    
    face_tensor_face_rec = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    
    curr_start_time = perf_counter()
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #获取视频总帧数

    start_frame_number = 0
    frame_count=int(length*0.1)#10帧抽一帧？
    frame_jump = 5 #int(frame_count/5)
    start_frame_number = 0

    loop = 0
    #count_mtcn=0
    #count_blaze=0
    count_face_rec = 0
    
    while cap.isOpened() and loop<frame_count:
        loop+=1
        success, frame = cap.read()#frame表示截取到某一帧的图像
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        
        if success:                    
            face_rec,count = face_face_rec(frame, face_tensor_face_rec)
            
            if len(face_rec) and count>0:
                kontrol = count_face_rec+count
                for f in face_rec:
                    if count_face_rec<=kontrol and (count_face_rec<29):#29是人脸数？人脸提取的函数中定义最多提取5个人脸，怎么理解
                        face_tensor_face_rec[count_face_rec] = f
                        count_face_rec+=1

            start_frame_number+=frame_jump
    
    #store_rec= face_tensor_face_rec[:count_face_rec]

    dfdc_tensor=face_tensor_face_rec[:count_face_rec]
    
    dfdc_tensor = torch.tensor(dfdc_tensor, device=device).float()

    # Preprocess the images.
    dfdc_tensor = dfdc_tensor.permute((0, 3, 1, 2))

    for i in range(len(dfdc_tensor)):
        dfdc_tensor[i] = normalize_transform(dfdc_tensor[i] / 255.)
    
    # the tranformer accepts batch of <=32.
    if not len(non_empty(dfdc_tensor, df_len=-1, lower_bound=-1, upper_bound=-1, flag=False)):
        return torch.tensor(0.5).item()
        
    dfdc_tensor = dfdc_tensor.contiguous()#强拷贝，断开二者之间的关系，y=x.contiguous(),即改变y的值也不会影响x
    df_len = len(dfdc_tensor)#代表什么？32-64-90
    print('wai')
    with torch.no_grad():         
        dfdc_tensor = dfdc_tensor.to(device)
        thrtw =32
        if df_len<33:
            thrtw =df_len  
        y_predCViT = model(dfdc_tensor[0:thrtw])
        
        if df_len>32:
            dft = non_empty(dfdc_tensor, df_len, lower_bound=32, upper_bound=64, flag=True)
            if len(dft):
                dft = torch.tensor(dft, device=device)
                y_predCViT = pred_tensor(y_predCViT, model(dft))
        if df_len>64:
            dft = non_empty(dfdc_tensor, df_len, lower_bound=64, upper_bound=90, flag=True)
            if len(dft):
                dft = torch.tensor(dft, device=device)
                y_predCViT = pred_tensor(y_predCViT, model(dft))
        print(pred_sig(y_predCViT))
        if trainonDFDC:
            decCViT = pre_process_prediction(pred_sig(y_predCViT)) #DFDC训练的模型，有两个概率值
        else:
            decCViT = custom_video_round(pred_sig(y_predCViT))   #报错0703  
        print('transfuse', filename, "Prediction:",decCViT.item())
        return decCViT.item()

def non_empty(dfdc_tensor, df_len, lower_bound, upper_bound, flag):
    
    thrtw=df_len
    if df_len>=upper_bound:
        thrtw=upper_bound
        
    if flag==True:
        return dfdc_tensor[lower_bound:thrtw]
    elif flag==False:
        return dfdc_tensor
        
    return []
    
def pred_sig(dfdc_tensor):
    return torch.sigmoid(dfdc_tensor.squeeze())

def pred_tensor(dfdc_tensor, pre_tensor):
    return torch.cat((dfdc_tensor,pre_tensor),0)

def pre_process_prediction(y_pred):
    f=[]
    r=[]
    if len(y_pred)>2:
        for i, j in y_pred:
            f.append(i)
            r.append(j)
        f_c = sum(f)/len(f)
        r_c= sum(r)/len(r)
        if f_c>r_c:
            return f_c
        else:
            r_c = abs(1-r_c)
            return r_c
    else:
        return torch.tensor(0.5)
    
#=======一个视频中只要有一帧为假，这个视频即被判断为假。此段代码来自crossefficient=这里应用有问题，原代码里该函数是处理多个人脸情况，有任一人脸大于0.55则该视频判断为假==
def custom_video_round(preds):
    #print('custom_video_round')    
    for pred_value in preds:
        if pred_value > 0.55:
            return pred_value
    '''f=[]
    for i in preds:
        f.append(i)
    f_c = sum(f)/len(f)
    return f_c'''
    return mean(preds)


#==========


def getlabel(videodir,metadata):#获得测试视频对应的真实label
    framefiles = os.listdir(videodir)
    labellist = []
    for framefile in framefiles:
        if framefile[-4:] == ".mp4":
            if data[framefile]['label']=='REAL':
                labellist.append(0)
            elif data[framefile]['label']=='FAKE':
                labellist.append(1)

    return labellist

def real_or_fake(filenames, predictions): #概率值转变为REAL/FAKE，并计算准确个数
    j=0
    correct = 0
    label="REAL"
    for i in filenames:
        if data[i]['label'] == 'REAL' and predictions[j]<0.5:
            correct+=1
            label="REAL"
        if data[i]['label'] == 'FAKE' and predictions[j]>=0.5:
            correct+=1
            label="FAKE"
        
        print('Filname:',i, label)
        j+=1
        
    return correct

def custom_round(values):#概率值转变为0/1
    result = []
    for value in values:
        if value > 0.5:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result) #result

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

  model_auc = auc(fpr, tpr)

  plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_"  + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()



start_time = perf_counter()
predictions = predict_on_video(filenames, num_workers=1)  #num_workers从4改为1就能跑通了
print(predictions)
end_time = perf_counter()
print("--- %s seconds ---" % (end_time - start_time))
predictlist = custom_round(predictions)


##metrics
'''
# for testing DFDC dataset
metafile = sample+'metadata.json'
if os.path.isfile(metafile):
    with open(metafile) as data_file:
        data = json.load(data_file)
correct_labellist = getlabel(sample,data)

'''
#对于FF++数据集，所有子集的label都是fake，不需要metadata,celeb-df也可以
print(sample)
correct_labellist = []
if 'youtube' or 'real' in sample:
    #correct_labellist = []
    for i in range(len(os.listdir(sample))):
        correct_labellist.append(0)
else:
    print('ok')
    for i in range(len(os.listdir(sample))):
        correct_labellist.append(1)
#'''
print(correct_labellist)


loss_fn = torch.nn.BCEWithLogitsLoss()#计算损失，输入需要tensor型
tensor_labels = torch.tensor([float(label) for label in correct_labellist])#还要将数值转换为float
tensor_preds = torch.tensor([float(predss) for predss in predictlist])#计算损失时不受用的是概率值
loss = loss_fn(tensor_preds, tensor_labels).numpy()

#correct = real_or_fake(filenames, predictions)
#accuracy = correct/len(filenames)*100
accuracy2 = accuracy_score(predictlist, correct_labellist)

f1 = f1_score(correct_labellist, predictlist)


model_name = 'transfuse_res_RGBSRM'
print(model_name, "Test Accuracy:", accuracy2, "Loss:", loss, "F1", f1)
save_roc_curves(correct_labellist, predictlist, model_name, accuracy2, loss, f1)

'''
def real_or_fake(filenames, predictions): 
    j=0
    correct = 0
    label="REAL"
    for i in filenames:
        if predictions[j]<0.5:
            label="REAL"
        if predictions[j]>=0.5:
            label="FAKE"
        
        print('Filname:',i,label)
        j+=1

real_or_fake(filenames, predictions)#预测值转变为预测的真假
submission_dfcvit_nov16 = pd.DataFrame({"filename": filenames, "label": predictions})
submission_dfcvit_nov16.to_csv("cvit_predictions1.csv", index=False)
'''