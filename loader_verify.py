# from loader import NTIRELoaderCV2


# trainset = NTIRELoaderCV2(img_dir='/home/pegasus/DATA/Nikhil/LLIE/NTIRE24_Challenges/night-photography-rendering/val2/', task='train')

# print(len(trainset))

# print(trainset[0][0].shape)



import os 

listt1 = os.listdir('/home/pegasus/DATA/Nikhil/LLIE/NTIRE24_Challenges/night-photography-rendering/train_gt')
listt2= os.listdir('/home/pegasus/DATA/Nikhil/LLIE/NTIRE24_Challenges/night-photography-rendering/train_inp')

print(listt1)
print(listt2)

