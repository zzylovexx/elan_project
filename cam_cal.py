import numpy as np 
intrinsic=np.array([1418.667,0,640,0,1418.66775,360,0,0,1])
intrinsic=np.reshape(intrinsic,(3,3))
extrinsic=np.array([0.9988,0,0.04832,0,0,1,0,0,-0.04832,0,0.9988,1.33])
extrinsic=np.reshape(extrinsic,(3,4))
print(intrinsic)
print(extrinsic)
project=np.dot(intrinsic,extrinsic)
print(project)