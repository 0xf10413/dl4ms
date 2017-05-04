/*
 * mainwindow.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "mainwindow.h"
#include "tools.h"
#include <QDebug>


MainWindow::MainWindow(QWidget *parent) : QWidget(parent)
{
  m_glob_layout = new QHBoxLayout();

  /* Définie avant le pyeditor */
  m_outputScroll = new FOutputScroll("<Python output>\n");

  /* OpenGL + textEdit */
  QVBoxLayout *vbox_left = new QVBoxLayout;
  {
    QHBoxLayout *tmpLayout = new QHBoxLayout();

    /* OpenGL + boutons */
    {
      m_displayWidget = new DisplayWidget(this);
      m_play_pause = new QPushButton ("Pause");
      m_stop = new QPushButton ("Stop");
      m_faster = new QPushButton("Faster");
      m_slower = new QPushButton ("Slower");

      QLayout *displayButtonsLayout = new QHBoxLayout;
      displayButtonsLayout->addWidget(m_play_pause);
      displayButtonsLayout->addWidget(m_stop);
      displayButtonsLayout->addWidget(m_faster);
      displayButtonsLayout->addWidget(m_slower);

      QVBoxLayout *displayWidgetLayout = new QVBoxLayout;
      displayWidgetLayout->addWidget(m_displayWidget);
      displayWidgetLayout->addLayout(displayButtonsLayout);

      tmpLayout->addLayout(displayWidgetLayout);
    }

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
    m_launchLoadConstraints = new QPushButton("Charger la fonction de contrainte");
    m_launchConstraints = new QPushButton("Appliquer les contraintes");

    tmpLayout->addWidget(m_launchLoadFootstepper);
    tmpLayout->addWidget(m_launchLoadNetwork);
    tmpLayout->addWidget(m_launchLoadConstraints);
    tmpLayout->addWidget(m_launchConstraints);
    m_groupActions->setLayout(tmpLayout);
  }

  /* On lie le tout */
  m_glob_layout->addLayout(vbox_left);
  m_glob_layout->addWidget(m_groupActions);


  /* Liens divers */
  connect(m_pyEdit, &FPyEditor::dataMayHaveChanged,
      m_displayWidget, &DisplayWidget::refreshDataToPrint);

  connect(m_play_pause, &QPushButton::clicked,
      m_displayWidget, &DisplayWidget::play_pause);
  connect(m_stop, &QPushButton::clicked,
      m_displayWidget, &DisplayWidget::stop);
  connect(m_faster, &QPushButton::clicked,
      m_displayWidget, &DisplayWidget::faster);
  connect(m_slower, &QPushButton::clicked,
      m_displayWidget, &DisplayWidget::slower);

  connect(m_launchLoadFootstepper, &QPushButton::clicked,
      [this]{ m_pyEdit->launchPython(pythonFromRc(":/python/load_footstepper.py"));
      m_launchLoadFootstepper->setText("Recharger le footstepper"); });
  connect(m_launchLoadNetwork, &QPushButton::clicked,
      [this]{ m_pyEdit->launchPython(pythonFromRc(":/python/load_regressor.py"));
      m_launchLoadNetwork->setText("Recharger le réseau principal");});
  connect(m_launchLoadConstraints, &QPushButton::clicked,
      [this]{ m_pyEdit->launchPython(pythonFromRc(":/python/load_constraints.py"));
      m_launchLoadConstraints->setText("Recharger la fonction de contrainte");});
  connect(m_launchConstraints, &QPushButton::clicked,
      [this]{ m_pyEdit->launchPython(pythonFromRc(":/python/launch_constraints.py"));
      m_launchConstraints->setText("Recalculer les contraintes");});

  setLayout(m_glob_layout);
  resize(1000,1000);
  m_pyEdit->setFocus();

  /* Actualisation des données */
  m_displayWidget->refreshDataToPrint(*m_pyEdit);
}

