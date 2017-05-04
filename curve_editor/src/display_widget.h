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
#include <QMatrix4x4>

#include <numpy/arrayobject.h>

#include "transform3d.h"
#include "camera3d.h"

class QOpenGLShaderProgram;

class DisplayWidget : public QOpenGLWidget,
                      protected QOpenGLFunctions
{
  Q_OBJECT

private:
  QOpenGLBuffer m_curve_buf, m_skel_buf;
  QOpenGLVertexArrayObject m_curve_obj, m_skel_obj;
  QOpenGLShaderProgram *m_program;

  // Private Helpers
  void printVersionInformation();

  // Shader Information
  int u_modelToWorld;
  int u_worldToCamera;
  int u_cameraToView;
  QMatrix4x4 m_projection;
	Camera3D m_camera;
  Transform3D m_transform;

  // Render information
  int m_current_frame, m_fps;

protected Q_SLOTS:
  void update();
public Q_SLOTS:
  void refreshDataToPrint(class FPyEditor &); // Réévalue les données d'espace
  void play_pause();
  void stop();
  void faster();
  void slower();

protected:
	void keyPressEvent(QKeyEvent *ev);
	void keyReleaseEvent(QKeyEvent *ev);
	void mousePressEvent(QMouseEvent *ev);
	void mouseReleaseEvent(QMouseEvent *ev);

public:
  DisplayWidget(QWidget *parent = nullptr);
  ~DisplayWidget();
  void initializeGL();
  void resizeGL(int width, int height);
  void paintGL();
  void teardownGL();
} ;

#endif /* !DISPLAY_WIDGET_H */
