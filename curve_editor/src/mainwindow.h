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

#include "display_widget.h"

class MainWindow : public QWidget
{
  Q_OBJECT

private:
    QHBoxLayout *m_layout;
    QPlainTextEdit *m_textEdit;
    DisplayWidget *m_displayWidget;
public:
    MainWindow(QWidget *parent = nullptr);
};

#endif /* !MAINWINDOW_H */
