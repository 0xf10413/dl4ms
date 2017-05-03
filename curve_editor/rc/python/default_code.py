# Code python, ctrl+entrée pour valider
import sys
from Quaternions import Quaternions
from math import sin, cos

sys.path.append('../motion')

npz = np.load('/home/flodeu/X_regression.npz')
Xrecn = npz['Xrecn']#[0,3:-7,:]
Xrecn = np.swapaxes(Xrecn[0], 0, 1)
Xtraj = npz['curve'][0,...]
Xtraj = np.swapaxes(Xtraj,1,0)
joints, root_x, root_z, root_r = Xrecn[:,:-7], Xrecn[:,-7], Xrecn[:,-6], Xrecn[:,-5]
joints = joints.reshape((len(joints), -1, 3))
rotation = Quaternions.id(1)
offsets = []
translation = np.array([[0,0,0]])

nb_frame = 10
for i in range(nb_frame):#range(len(joints)):
    joints[i,:,:] = rotation * joints[i]
    joints[i,:,0] = joints[i,:,0] + translation[0,0]
    joints[i,:,2] = joints[i,:,2] + translation[0,2]
    rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
    offsets.append(rotation * np.array([0,0,1]))
    translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

#skel = joints
#skel_parents = np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20])
