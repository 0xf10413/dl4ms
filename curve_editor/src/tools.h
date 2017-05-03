/*
 * tools.h
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef TOOLS_H
#define TOOLS_H

#include <QFile>

inline QString pythonFromRc(QString rcPath)
{
  QString python_code;
  QFile file(rcPath);
  file.open(QIODevice::ReadOnly | QIODevice::Text);
  QTextStream in(&file);
  while(!in.atEnd())
    python_code += in.readLine() + "\n";

  return python_code;
}

#endif /* !TOOLS_H */
