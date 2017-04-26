import bpy, bmesh
import mathutils
import numpy as np
import os, sys
from math import sin, cos

sys.path.append(r'C:\Users\p1619885\Downloads\papier_code\dl4ms\motion')

from Quaternions import Quaternions

"""
Import de mouvement via numpy, insipré de AnimationPlot
"""

# Préparation de la scène
scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = 2000
bpy.context.scene.frame_set(frame = 1)


# En mode OBJECT
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
cur_obj = bpy.context.active_object

# Récupération des mouvements
npz = np.load(r'C:\Users\p1619885\python\curves\X_regression.npz')
Xrecn = npz['Xrecn']#[0,3:-7,:]
Xrecn = np.swapaxes(Xrecn[0], 0, 1)
Xtraj = npz['curve'][0,...]
Xtraj = np.swapaxes(Xtraj,1,0)

print(Xtraj.shape) # 7200 frames x 3 données de position (vx, vy, omega)
print(Xrecn.shape) # 3*22 joints x 7200 frames



joints, root_x, root_z, root_r = Xrecn[:,:-7], Xrecn[:,-7], Xrecn[:,-6], Xrecn[:,-5]
joints = joints.reshape((len(joints), -1, 3))

nb_frame = min(Xrecn.shape[0],scn.frame_end)
print("Animating for",nb_frame,"frames")

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
skel_offset = mathutils.Vector((1,1,1))
if ob is None:
    mesh = bpy.data.meshes.new('skeleton')
    ob = bpy.data.objects.new('skeleton', mesh)
    scn.objects.link(ob)
    scn.objects.active = ob
    ob.select = True
    mesh.from_pydata([(0,0,0)]*(22*4), 
     [(4*i,4*i+1) for i in range(22)],
     [(4*i, 4*i+1, 4*i+3, 4*i+2) for i in range(22)])
    mesh.update()

mesh = ob.data
vertices = mesh.vertices
action = bpy.data.actions.new('Animation')
mesh.animation_data_create()
mesh.animation_data.action = action

# Animation du squelette
for j in range(len(parents)):
    data_path = "vertices[%d].co"
    fcurves = [action.fcurves.new(data_path % (4*j), i) for i in range(3)]
    fcurves += [action.fcurves.new(data_path % (4*j+1), i) for i in range(3)]
    fcurves += [action.fcurves.new(data_path % (4*j+2), i) for i in range(3)]
    fcurves += [action.fcurves.new(data_path % (4*j+3), i) for i in range(3)]
    for f in range(nb_frame):
        if parents[j] != -1:
            vertices[4*j].co[0] = joints[f,j,0]
            vertices[4*j+1].co[0] = joints[f,parents[j],0]
            vertices[4*j].co[1] = joints[f,j,2]
            vertices[4*j+1].co[1] = joints[f,parents[j],2]
            vertices[4*j].co[2] = joints[f,j,1]
            vertices[4*j+1].co[2] = joints[f,parents[j],1]
            vertices[4*j+2].co = vertices[4*j].co + skel_offset
            vertices[4*j+3].co = vertices[4*j+1].co + skel_offset
        for fcu, val in zip(fcurves[:3], vertices[4*j].co):
            fcu.keyframe_points.insert(f, val, {'FAST'})
        for fcu, val in zip(fcurves[3:6], vertices[4*j+1].co):
            fcu.keyframe_points.insert(f, val, {'FAST'})
        for fcu, val in zip(fcurves[6:9], vertices[4*j+2].co):
            fcu.keyframe_points.insert(f, val, {'FAST'})
        for fcu, val in zip(fcurves[9:12], vertices[4*j+3].co):
            fcu.keyframe_points.insert(f, val, {'FAST'})

mesh.update()

# Génération de la courbe de déplacement
obc = scn.objects.get('path', None)
if obc is None:
    mesh = bpy.data.meshes.new('path')
    obc = bpy.data.objects.new('path', mesh)
    scn.objects.link(obc)
    scn.objects.active = obc
    obc.select = True
    mesh.from_pydata([(0,0,0)]*(nb_frame), [(i,i+1) for i in range(nb_frame-1)], [])
    mesh.update()
else:
    mesh = obc.data

vertices = mesh.vertices
prev_co = vertices[0].co
prev_theta = 0
for f in range(1,nb_frame):
    prev_theta += Xtraj[f,2]
    vx = Xtraj[f,0]
    vy = Xtraj[f,1]
    prev_co[0] += cos(prev_theta)*vx - sin(prev_theta)*vy
    prev_co[1] += sin(prev_theta)*vx + cos(prev_theta)*vy

    vertices[f].co[0] = prev_co[0]
    vertices[f].co[1] = prev_co[1]

