/*
 * foutput_scroll.h
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef FOUTPUT_SCROLL_H
#define FOUTPUT_SCROLL_H

#include <QLabel>
#include <QVBoxLayout>
#include <QScrollArea>

class FOutputScroll : public QScrollArea
{
  Q_OBJECT

private:
    QLabel *m_label;
    void scrollDown(); // Re-scroll en bas de l'output
public:
    FOutputScroll(const QString &defaultText=QString(),
        QWidget *parent = nullptr);
    void addText (const QString &);
    void addLine (const QString &s);
};

#endif /* !FOUTPUT_SCROLL_H */
