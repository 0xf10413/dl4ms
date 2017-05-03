/*
 * mainwindow.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "mainwindow.h"
#include <QDebug>

static const char *loadFootStepper =
/* Imports supplémentaires */
"from Network import Network\n"
"from network import create_core, create_regressor, create_footstepper\n"
"from constraints import constrain, foot_sliding,"
" joint_lengths, trajectory, multiconstraint\n"

/* Chargement des données */
"rng = np.random.RandomState(23455)\n"
"preprocess = np.load('../synth/preprocess_core.npz')\n"
"preprocess_footstepper = np.load('../synth/preprocess_footstepper.npz')\n"
"batchsize = 1\n"

/* Fonction de création du réseau */
"def create_network(window, input):\n"
"    network_first = create_regressor(batchsize=batchsize, window=window, input=input, dropout=0.0)\n"
"    network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw:x/2)\n"
"    network_second.load(np.load('../synth/network_core.npz'))\n"
"    network = Network(network_first, network_second[1], params=network_first.params)\n"
"    network.load(np.load('../synth/network_regression.npz'))\n"
"    return network_first, network_second, network\n"

/* Création du footstepper */
"input = theano.tensor.ftensor3()\n"
"Torig = curve\n"
"Torig = np.expand_dims(Torig, axis=0)\n"
"Torig = (Torig - preprocess['Xmean'][:,-7:-4]) / preprocess['Xstd'][:,-7:-4]\n"

"network_footstepper = create_footstepper(batchsize=batchsize, window=Torig.shape[0], dropout=0.0)\n"
"network_footstepper.load(np.load('../synth/network_footstepper.npz'))\n"
"network_footstepper_func = theano.function([input], network_footstepper(input), allow_input_downcast=True)\n"

"W = network_footstepper_func(Torig[:,:3])\n"

"alpha, beta = 1.0, 0.0\n"
"minstep, maxstep = 0.9, -0.5\n"
"off_lh, off_lt, off_rh, off_rt = 0.0, -0.1, np.pi+0.0, np.pi-0.1\n"
"Torig = (np.concatenate([Torig,\n"
"    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lh)>np.clip(np.cos(W[:,1:2])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,\n"
"    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lt)>np.clip(np.cos(W[:,2:3])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,\n"
"    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rh)>np.clip(np.cos(W[:,3:4])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,\n"
"    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rt)>np.clip(np.cos(W[:,4:5])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1], axis=1))\n"
;

static const char *loadNetwork =
"network_first, network_second, network = create_network(Torig.shape[2], Torig.shape[1])\n"
"network_func = theano.function([input], network(input), allow_input_downcast=True)\n"
"Xrecn = network_func(Torig)\n"
"Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']\n"
"Xrecn = np.swapaxes(Xrecn[0],1,0)\n"
"Xtraj = ((Torig * preprocess['Xstd'][:,-7:]) + preprocess['Xmean'][:,-7:]).copy()\n"
"joints, root_x, root_z, root_r = Xrecn[:,:-7], Xrecn[:,-7], Xrecn[:,-6], Xrecn[:,-5]\n"
"joints = joints.reshape((len(joints), -1, 3))\n"
;

MainWindow::MainWindow(QWidget *parent) : QWidget(parent)
{
  m_glob_layout = new QHBoxLayout();

  /* Définie avant le pyeditor */
  m_outputScroll = new FOutputScroll("<Python output>\n");

  /* OpenGL + textEdit */
  QVBoxLayout *vbox_left = new QVBoxLayout;
  {
    QLayout *tmpLayout = new QHBoxLayout();

    m_displayWidget = new DisplayWidget(this);
    tmpLayout->addWidget(m_displayWidget);

    m_pyEdit = new FPyEditor(m_outputScroll);
    tmpLayout->addWidget(m_pyEdit);

    vbox_left->addLayout(tmpLayout);
  }

  /* scrolltext */
  vbox_left->addWidget(m_outputScroll);
  vbox_left->setStretch(0,10);
  vbox_left->setStretch(1,5);

  /* Actions groupées */
  m_groupActions = new QGroupBox;
  {
    QVBoxLayout *tmpLayout = new QVBoxLayout;
    m_launchLoadFootstepper = new QPushButton("Charger le footstepper");
    m_launchLoadNetwork = new QPushButton("Charger le réseau principal");
    tmpLayout->addWidget(m_launchLoadFootstepper);
    tmpLayout->addWidget(m_launchLoadNetwork);
    tmpLayout->addWidget(new QPushButton("[nothing]"));
    m_groupActions->setLayout(tmpLayout);
  }

  /* On lie le tout */
  m_glob_layout->addLayout(vbox_left);
  m_glob_layout->addWidget(m_groupActions);



  /* Liens divers */
  connect(m_pyEdit, &FPyEditor::dataMayHaveChanged,
      m_displayWidget, &DisplayWidget::refreshDataToPrint);
  connect(m_launchLoadFootstepper, &QPushButton::clicked,
      [this]{ m_pyEdit->launchPython(loadFootStepper);
      m_launchLoadFootstepper->setText("Recharger le footstepper"); });
  connect(m_launchLoadNetwork, &QPushButton::clicked,
      [this]{ m_pyEdit->launchPython(loadNetwork);
      m_launchLoadNetwork->setText("Recharger le réseau principal");});

  setLayout(m_glob_layout);
  resize(1000,800);
  m_pyEdit->setFocus();

  /* Actualisation des données */
  m_displayWidget->refreshDataToPrint(*m_pyEdit);
}

