import bpy
import mathutils
import numpy as np
import os

"""
Petit exemple d'animation avec blender + python
"""

# Préparation de la scène
scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = 100
bpy.context.scene.frame_set(frame = 1)

# En mode POSE
if bpy.context.active_object.mode != 'POSE':
    raise RuntimeError("Not in Pose mode")

npz = np.load(r'C:\Users\p1619885\python\curves\X_regression.npz')
Xrecn = npz['Xrecn']

print(Xrecn.shape)

for i in range(1,100):
    edit_bones = bpy.context.object.pose.bones['Hips']
    edit_bones.location.x = i
    bpy.ops.anim.keyframe_insert(type="Location")
    bpy.context.scene.frame_set(frame = i)