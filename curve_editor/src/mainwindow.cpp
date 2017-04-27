/*
 * mainwindow.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "mainwindow.h"
#include <QPushButton>

MainWindow::MainWindow(QWidget *parent) : QWidget(parent)
{
  m_layout = new QHBoxLayout();

  m_displayWidget = new DisplayWidget(this);
  m_layout->addWidget(m_displayWidget);

  m_textEdit = new QPlainTextEdit(this);
  m_layout->addWidget(m_textEdit);

  setLayout(m_layout);
  resize(500,500);
}

