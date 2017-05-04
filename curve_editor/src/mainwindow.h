/*
 * mainwindow.h
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QVBoxLayout>

#include "display_widget.h"
#include "fpyeditor.h"
#include "foutput_scroll.h"

class MainWindow : public QWidget
{
  Q_OBJECT

private:
    QHBoxLayout *m_glob_layout;
    QGroupBox *m_groupActions;

    QPushButton *m_launchLoadFootstepper;
    QPushButton *m_launchLoadNetwork;
    QPushButton *m_launchLoadConstraints;
    QPushButton *m_launchConstraints;

    QPushButton *m_play_pause, *m_stop,
                *m_faster, *m_slower;

    FPyEditor *m_pyEdit;
    DisplayWidget *m_displayWidget;
    FOutputScroll *m_outputScroll;
public:
    MainWindow(QWidget *parent = nullptr);
    void paintEvent(QPaintEvent *)
    {}
};

#endif /* !MAINWINDOW_H */
