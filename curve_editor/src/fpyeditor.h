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
#include <numpy/arrayobject.h>
#include "foutput_scroll.h"

class FPyEditor : public QPlainTextEdit
{
  Q_OBJECT

private:
  FOutputScroll *m_output;
  boost::python::object m_main_module, m_main_ns;
  boost::python::object m_orig_ns; // Pour retrouver les objects créés
  std::string handle_pyerror();

Q_SIGNALS:
  /* Indique un changement des données python en background */
  void dataMayHaveChanged(FPyEditor &);

public:
  FPyEditor(FOutputScroll *output, QWidget *parent = nullptr);
  void launchPython(const QString &code, bool mute=false);
  void keyPressEvent(QKeyEvent *e);

  friend class DisplayWidget;
};

#endif /* !F_PYEDITOR_H */
