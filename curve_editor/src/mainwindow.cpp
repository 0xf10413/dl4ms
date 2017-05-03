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
      [this]{ m_pyEdit->launchPython(pythonFromRc(":/python/load_footstepper.py"));
      m_launchLoadFootstepper->setText("Recharger le footstepper"); });
  connect(m_launchLoadNetwork, &QPushButton::clicked,
      [this]{ m_pyEdit->launchPython(pythonFromRc(":/python/load_regressor.py"));
      m_launchLoadNetwork->setText("Recharger le réseau principal");});

  setLayout(m_glob_layout);
  resize(1000,800);
  m_pyEdit->setFocus();

  /* Actualisation des données */
  m_displayWidget->refreshDataToPrint(*m_pyEdit);
}

