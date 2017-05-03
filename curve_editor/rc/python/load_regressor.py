#!/usr/bin/env python3
"""
Chargement du régresseur.
C'est le réseau principal, il contient l'encodeur et le feed-forward
"""

from Quaternions import Quaternions


network_first, network_second, network = create_network(Torig.shape[2], Torig.shape[1])
network_func = theano.function([input], network(input), allow_input_downcast=True)
Xrecn = network_func(Torig)
Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
Xrecn = np.swapaxes(Xrecn[0],0,1)
Xtraj = ((Torig * preprocess['Xstd'][:,-7:]) + preprocess['Xmean'][:,-7:]).copy()
joints, root_x, root_z, root_r = Xrecn[:,:-7], Xrecn[:,-7], Xrecn[:,-6], Xrecn[:,-5]
joints = joints.reshape((len(joints), -1, 3))


rotation = Quaternions.id(1)
offsets = []
translation = np.array([[0,0,0]])

print(joints.shape)
joints_bkp = joints.copy()
joints_true_bkp = np.copy(joints)
skel_bkp = skel

nb_frame = len(joints)

for i in range(nb_frame):
    joints[i,:,:] = rotation * joints[i]
    joints[i,:,0] = joints[i,:,0] + translation[0,0]
    joints[i,:,2] = joints[i,:,2] + translation[0,2]
    rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
    offsets.append(rotation * np.array([0,0,1]))
    translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

joints = joints.astype(np.float32)
skel = joints
