/*
 * pyeditor.h
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef F_PYEDITOR_H
#define F_PYEDITOR_H

#include <QPlainTextEdit>
#include <QLabel>
#include <boost/python/object.hpp>
#include "foutput_scroll.h"

class FPyEditor : public QPlainTextEdit
{
  Q_OBJECT
private:
  FOutputScroll *m_output;
  boost::python::object m_main_module, m_main_ns;
  std::string handle_pyerror();
public:
  FPyEditor(FOutputScroll *output, QWidget *parent = nullptr);
  void launchPython(const QString &);
  void keyPressEvent(QKeyEvent *e);
};

#endif /* !F_PYEDITOR_H */
