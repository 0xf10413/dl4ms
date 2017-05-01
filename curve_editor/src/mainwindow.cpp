/*
 * mainwindow.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "mainwindow.h"
#include <QDebug>


MainWindow::MainWindow(QWidget *parent) : QWidget(parent)
{
  m_glob_layout = new QVBoxLayout();

  /* Définie avant le pyeditor */
  m_outputScroll = new FOutputScroll("<Python output>\n");


  /* Ajout de la première ligne : OpenGL + textedit */
  m_layout = new QHBoxLayout();

  m_displayWidget = new DisplayWidget(this);
  m_layout->addWidget(m_displayWidget);

  m_pyEdit = new FPyEditor(m_outputScroll);
  m_layout->addWidget(m_pyEdit);

  m_glob_layout->addLayout(m_layout);


  /* Ajout de la deuxième ligne : scrolltext */
  m_glob_layout->addWidget(m_outputScroll);
  m_glob_layout->setStretch(0,10);
  m_glob_layout->setStretch(1,5);

  /* Liens divers */
  connect(m_pyEdit, &FPyEditor::dataMayHaveChanged,
      m_displayWidget, &DisplayWidget::refreshDataToPrint);

  setLayout(m_glob_layout);
  resize(800,500);
  m_pyEdit->setFocus();
}

