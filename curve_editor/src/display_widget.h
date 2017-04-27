/*
 * display_widget.h
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef DISPLAY_WIDGET_H
#define DISPLAY_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>

class QOpenGLShaderProgram;

class DisplayWidget : public QOpenGLWidget,
                      protected QOpenGLFunctions
{
  Q_OBJECT

private:
  QOpenGLBuffer m_vertex;
  QOpenGLVertexArrayObject m_object;
  QOpenGLShaderProgram *m_program;

  // Private Helpers
  void printVersionInformation();

public:
  DisplayWidget(QWidget *parent = nullptr);
  void initializeGL();
  //void resizeGL(int width, int height);
  void paintGL();
  void teardownGL();
} ;

#endif /* !DISPLAY_WIDGET_H */
