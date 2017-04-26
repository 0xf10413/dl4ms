import bpy
import mathutils
import numpy as np
import os, sys

"""
Import de mouvement via numpy
"""

# Préparation de la scène
scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = 100
bpy.context.scene.frame_set(frame = 1)


# En mode POSE
if bpy.context.active_object.mode != 'POSE':
    bpy.ops.object.mode_set(mode='POSE')
cur_obj = bpy.context.active_object

# Récupération des mouvements
npz = np.load(r'C:\Users\p1619885\python\curves\X_regression.npz')
Xrecn = npz['Xrecn'][0,3:-7,:]

# Récupération des articulations effectivement utilisées
filt = np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])
mask = np.ones(shape=(31,),dtype=bool)
mask[filt] = False


joints_name = cur_obj.data.bones.keys()
joints_name = np.array(joints_name)

for to_del in joints_name[mask]:
    edit_bones = cur_obj.pose.bones[to_del]
    print(to_del)

joints_name = joints_name[filt]

#print(Xrecn.shape) # 3*21 joints x 7200 frames
#print(joints_name.shape) # 21 noms

nb_frame = Xrecn.shape[1]
nb_frame = 100

for f in range(nb_frame):
    for i in range(21):
        edit_bones = cur_obj.pose.bones[joints_name[i]]
        edit_bones.location.x = Xrecn[3*i,f]
        edit_bones.location.y = Xrecn[3*i+1,f]
        edit_bones.location.z = Xrecn[3*i+2,f]
        edit_bones.keyframe_insert(data_path='location', frame=f)

#for i in range(1,100):
#    edit_bones = bpy.context.object.pose.bones['Hips']
#    edit_bones.location.x = i
#    bpy.ops.anim.keyframe_insert(type="Location")
#    bpy.context.scene.frame_set(frame = i)