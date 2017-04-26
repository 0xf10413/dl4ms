import bpy, bmesh
import mathutils
import numpy as np
import os, sys

sys.path.append(r'C:\Users\p1619885\Downloads\papier_code\dl4ms\motion')

from Quaternions import Quaternions

"""
Import de mouvement via numpy, insipré de AnimationPlot
"""

# Préparation de la scène
scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = 100
bpy.context.scene.frame_set(frame = 1)


# En mode OBJECT
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
cur_obj = bpy.context.active_object

# Récupération des mouvements
npz = np.load(r'C:\Users\p1619885\python\curves\X_regression.npz')
Xrecn = npz['Xrecn']#[0,3:-7,:]
Xrecn = np.swapaxes(Xrecn[0], 0, 1)

print(Xrecn.shape) # 3*22 joints x 7200 frames

nb_frame = Xrecn.shape[1]

joints, root_x, root_z, root_r = Xrecn[:,:-7], Xrecn[:,-7], Xrecn[:,-6], Xrecn[:,-5]
joints = joints.reshape((len(joints), -1, 3))

rotation = Quaternions.id(1)
offsets = []
translation = np.array([[0,0,0]])

print(joints.shape)

for i in range(nb_frame):#range(len(joints)):
    joints[i,:,:] = rotation * joints[i]
    joints[i,:,0] = joints[i,:,0] + translation[0,0]
    joints[i,:,2] = joints[i,:,2] + translation[0,2]
    rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
    offsets.append(rotation * np.array([0,0,1]))
    translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

parents = np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20])

# Génération d'un nouveeau squelette si besoin
ob = scn.objects.get('skeleton', None)
if ob is None:
    mesh = bpy.data.meshes.new('skeleton')
    ob = bpy.data.objects.new('skeleton', mesh)
    scn.objects.link(ob)
    scn.objects.active = ob
    ob.select = True
    mesh.from_pydata([(0,0,0)]*(22*2), [(2*i,2*i+1) for i in range(22)], [])
    mesh.update()

mesh = ob.data
vertices = mesh.vertices
action = bpy.data.actions.new('Animation')
mesh.animation_data_create()
mesh.animation_data.action = action

# Animation du squelette
for j in range(len(parents)):
    data_path = "vertices[%d].co"
    fcurves = [action.fcurves.new(data_path % (2*j), i) for i in range(3)]
    fcurves += [action.fcurves.new(data_path % (2*j+1), i) for i in range(3)]
    for f in range(nb_frame):
        if parents[j] != -1:
            vertices[2*j].co[0] = joints[f,j,0]
            vertices[2*j+1].co[0] = joints[f,parents[j],0]
            vertices[2*j].co[1] = joints[f,j,2]
            vertices[2*j+1].co[1] = joints[f,parents[j],2]
            vertices[2*j].co[2] = joints[f,j,1]
            vertices[2*j+1].co[2] = joints[f,parents[j],1]
        for fcu, val in zip(fcurves[:3], vertices[2*j].co):
            fcu.keyframe_points.insert(f, val, {'FAST'})
        for fcu, val in zip(fcurves[3:], vertices[2*j+1].co):
            fcu.keyframe_points.insert(f, val, {'FAST'})

mesh.update()   


