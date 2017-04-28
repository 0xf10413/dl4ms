/*
 * foutput_scroll.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "foutput_scroll.h"


FOutputScroll::FOutputScroll(const QString &defaultText,
    QWidget *parent) : QScrollArea(parent)
{
  m_label = new QLabel(defaultText);
  setWidgetResizable(true);
  setWidget(m_label);
  ensureWidgetVisible(m_label);
}

void FOutputScroll::addText (const QString &s)
{
  m_label->setText(m_label->text()+s);
  scrollDown();
}

void FOutputScroll::addLine (const QString &s)
{
  m_label->setText(m_label->text() + "\n >>> " + s);
  scrollDown();
}

void FOutputScroll::scrollDown()
{
  m_label->resize(widget()->sizeHint());
  ensureVisible(m_label->width()+100,
      m_label->height()+1000,0,0);
}
