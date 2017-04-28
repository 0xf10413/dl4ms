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
#include <QHBoxLayout>
#include <QVBoxLayout>

#include "display_widget.h"
#include "fpyeditor.h"
#include "foutput_scroll.h"

class MainWindow : public QWidget
{
  Q_OBJECT

private:
    QVBoxLayout *m_glob_layout;
    QHBoxLayout *m_layout;

    FPyEditor *m_pyEdit;
    DisplayWidget *m_displayWidget;
    FOutputScroll *m_outputScroll;
public:
    MainWindow(QWidget *parent = nullptr);
    void paintEvent(QPaintEvent *)
    {}
};

#endif /* !MAINWINDOW_H */
