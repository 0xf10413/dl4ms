# Code python, ctrl+entrée pour valider
index = 30
length = 15*480
npz = np.load('/home/flodeu/X_regression.npz')
Xrecn = npz['Xrecn']
Xrecn = np.swapaxes(Xrecn[0], 0, 1)
Xtraj = npz['curve'][0,...]
#Xtraj = np.swapaxes(Xtraj, 1, 0)
curve = Xtraj

#joints = Xrecn[:,:-7]
joints = joints_from_Xrecn(Xrecn)
joints = joints.reshape(len(joints), -1, 3)
