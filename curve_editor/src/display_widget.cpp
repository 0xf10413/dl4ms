/*
 * display_widget.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "display_widget.h"
#include "fpyeditor.h"
#include "vertex.hpp"
#include "input.h"

#include <QDebug>
#include <QString>
#include <QKeyEvent>
#include <QOpenGLShaderProgram>

#include <numpy/arrayobject.h>

#include <array>
#include <random>

static std::array<Vertex,7200> curve_vertices { };
static std::array<Vertex,2*22> skel_vertices { };

void DisplayWidget::initializeGL()
{
  // Initialize OpenGL Backend
  initializeOpenGLFunctions();
  connect(this, &DisplayWidget::frameSwapped, this, &DisplayWidget::update);
  printVersionInformation();

  // Set global information
  glEnable(GL_CULL_FACE);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  // Application-specific initialization
  {
    // Create Shader (Do not release until VAO is created)
    m_program = new QOpenGLShaderProgram();
    m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/simple.vert");
    m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/simple.frag");
    m_program->link();
    m_program->bind();

    // Cache Uniform Locations
    u_modelToWorld = m_program->uniformLocation("modelToWorld");
    u_worldToCamera = m_program->uniformLocation("worldToCamera");
    u_cameraToView = m_program->uniformLocation("cameraToView");

    // Create Buffer for curve (Do not release until VAO is created)
    m_curve_buf.create();
    m_curve_buf.bind();
    m_curve_buf.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    m_curve_buf.allocate(curve_vertices.data(), curve_vertices.size()*sizeof(Vertex));

    // Create corresponding Vertex Array Object
    m_curve_obj.create();
    m_curve_obj.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, Vertex::positionOffset(), Vertex::PositionTupleSize, Vertex::stride());
    m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());
    m_curve_obj.release();

    // Create Buffer for skeleton (Do not release until VAO is created)
    m_skel_buf.create();
    m_skel_buf.bind();
    m_skel_buf.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    m_skel_buf.allocate(skel_vertices.data(), skel_vertices.size()*sizeof(Vertex));

    // Create corresponding Vertex Array Object
    m_skel_obj.create();
    m_skel_obj.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, Vertex::positionOffset(), Vertex::PositionTupleSize, Vertex::stride());
    m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());

    // Release (unbind) all
    m_skel_buf.release();
    m_program->release();
  }
}

DisplayWidget::DisplayWidget(QWidget *parent) : QOpenGLWidget(parent)
{
  m_transform.translate(.0, -1.0, -5.0f);
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  setFormat(format);

  setMinimumSize(QSize(400,400));
  setMaximumSize(QSize(400,400));
  curve_vertices[0] = Vertex(QVector3D{1,1,1}, QVector3D{1,0,0} );
  curve_vertices[1] = Vertex(QVector3D{1,0,0}, QVector3D{0,1,0} );
  curve_vertices[2] = Vertex(QVector3D{1,0,0}, QVector3D{0,1,0} );
  curve_vertices[3] = Vertex(QVector3D{0,1,0}, QVector3D{0,0,1} );
  curve_vertices[4] = Vertex(QVector3D{0,0,1}, QVector3D{1,0,1} );

  std::uniform_real_distribution<float> unif(0,1);
  std::mt19937 mersenne(41);

  /* Random colors */
  for (size_t i = 0; i < curve_vertices.size(); ++i)
  {
    float r = (float)i/curve_vertices.size();
    float g = (1.f-i)/curve_vertices.size();
    curve_vertices[i].setColor({r, g, 1});
  }
}

DisplayWidget::~DisplayWidget()
{
  teardownGL();
}

void DisplayWidget::paintGL()
{
  // Clear
  glClear(GL_COLOR_BUFFER_BIT);

  // Render using our shader
  m_program->bind();
  m_program->setUniformValue(u_worldToCamera, m_camera.toMatrix());
  m_program->setUniformValue(u_cameraToView, m_projection);
  {
    m_curve_obj.bind();
    m_program->setUniformValue(u_modelToWorld, m_transform.toMatrix());
    glDrawArrays(GL_LINE_STRIP, 0, curve_vertices.size()*sizeof(Vertex));
    m_curve_obj.release();

    m_skel_obj.bind();
    m_program->setUniformValue(u_modelToWorld, m_transform.toMatrix());
    glDrawArrays(GL_LINES, 0, skel_vertices.size()*sizeof(Vertex));
    m_skel_obj.release();
  }
  m_program->release();
}

void DisplayWidget::teardownGL()
{
  // Actually destroy our OpenGL information
  m_curve_obj.destroy();
  m_skel_obj.destroy();
  m_skel_buf.destroy();
  m_curve_buf.destroy();
  delete m_program;
}

void DisplayWidget::resizeGL(int width, int height)
{
  m_projection.setToIdentity();
  m_projection.perspective(45., width/float(height), .0, 1000.);
}

void DisplayWidget::printVersionInformation()
{
  QString glType;
  QString glVersion;
  QString glProfile;

  // Get Version Information
  glType = (context()->isOpenGLES()) ? "OpenGL ES" : "OpenGL";
  glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));

  // Get Profile Information
#define CASE(c) case QSurfaceFormat::c: glProfile = #c; break
  switch (format().profile())
  {
    CASE(NoProfile);
    CASE(CoreProfile);
    CASE(CompatibilityProfile);
  }
#undef CASE

  // qPrintable() will print our QString w/o quotes around it.
  qDebug() << qPrintable(glType) << qPrintable(glVersion) << "(" << qPrintable(glProfile) << ")";
}

void DisplayWidget::update()
{
  // Update input
  Input::update();

  // Camera Transformation
  if (Input::buttonPressed(Qt::LeftButton))
  {
    setFocus();
    static const float transSpeed = 0.1f;
    static const float rotSpeed   = 0.5f;

    // Handle rotations
    m_camera.rotate(-rotSpeed * Input::mouseDelta().x(), Camera3D::LocalUp);
    m_camera.rotate(-rotSpeed * Input::mouseDelta().y(), m_camera.right());

    // Handle translations
    QVector3D translation;
    if (Input::keyPressed(Qt::Key_Z))
      translation += m_camera.forward();

    if (Input::keyPressed(Qt::Key_S))
      translation -= m_camera.forward();

    if (Input::keyPressed(Qt::Key_Q))
      translation -= m_camera.right();

    if (Input::keyPressed(Qt::Key_D))
      translation += m_camera.right();

    if (Input::keyPressed(Qt::Key_A))
      translation -= m_camera.up();

    if (Input::keyPressed(Qt::Key_E))
      translation += m_camera.up();

    m_camera.translate(transSpeed * translation);
  }

  // Update instance information
  //m_transform.rotate(1.0f, QVector3D(0.0f, 1.0f, 0.0f));

  // Schedule a redraw
  QOpenGLWidget::update();
}

void DisplayWidget::keyPressEvent(QKeyEvent *event)
{
  if (event->isAutoRepeat())
  {
    event->ignore();
  }
  else
  {
    Input::registerKeyPress(event->key());
  }
}

void DisplayWidget::keyReleaseEvent(QKeyEvent *event)
{
  if (event->isAutoRepeat())
  {
    event->ignore();
  }
  else
  {
    Input::registerKeyRelease(event->key());
  }
}

void DisplayWidget::mousePressEvent(QMouseEvent *event)
{
  Input::registerMousePress(event->button());
}

void DisplayWidget::mouseReleaseEvent(QMouseEvent *event)
{
  Input::registerMouseRelease(event->button());
}

void DisplayWidget::refreshDataToPrint(FPyEditor &e)
{
  /* Tracé de la courbe, attention, elle est discrétisée */
  /* format : (frames * (dx,dy,d_omega) */
  {
    boost::python::object curve_ = e.m_main_ns["curve"];
    PyArrayObject *curve = reinterpret_cast<PyArrayObject*>(curve_.ptr());
    const int dims = PyArray_NDIM(curve);
    if (dims > 2)
    {
      qDebug() << "Error in " << __func__ << " : too many dimensions (" << dims << " > 2)";
      return;
    }

    const long dimi = PyArray_DIM(curve, 0), dimj = PyArray_DIM(curve, 1);
    if (dimi != 3)
    {
      qDebug() << "Error in " << __func__ << " : wrong 2nd dimension (" << dimi << " != 3)";
      return;
    }

    if (dimj > (signed long)curve_vertices.size())
    {
      qDebug() << "Error in " << __func__ <<
        " : not enough points allocated ( " << dimj << " > "
        << curve_vertices.size() << ")";
      return;
    }

    std::array<double,3> cur_pos {0.,0.,0.};
    curve_vertices[0].setPosition({0.f, 0.f, 0.f});
    curve_vertices[0].setColor({1, 1, 0});
    double *linex = reinterpret_cast<double*>(PyArray_GETPTR1(curve, 0));
    double *liney = reinterpret_cast<double*>(PyArray_GETPTR1(curve, 1));
    double *linez = reinterpret_cast<double*>(PyArray_GETPTR1(curve, 2));
    float prev_theta = 0.;
    for (int j = 1; j < dimj; ++j)
    {
      float b = (float)j/dimj;
      float vx = (float) linex[j];
      float vy = (float) liney[j];

      cur_pos[0] += cos(prev_theta)*vx - sin(prev_theta)*vy;
      cur_pos[1] += sin(prev_theta)*vx + cos(prev_theta)*vy;
      prev_theta += linez[j];
      curve_vertices[j].setPosition({(float)cur_pos[0],
          0.f, (float)cur_pos[1]
          });
      curve_vertices[j].setColor({1, 1, b});
    }
    for (size_t j = dimj; j < curve_vertices.size(); ++j)
    {
      curve_vertices[j].setPosition({(float)(cur_pos[0]),
          (float)(cur_pos[1]),
          (float)(cur_pos[2])});
      curve_vertices[j].setColor({1, 1, 1});
    }

    m_curve_buf.bind();
    m_curve_buf.write(0, curve_vertices.data(), curve_vertices.size()*sizeof(Vertex));
    m_curve_buf.release();
  }

  /* Tracé du squelette */
  {
    boost::python::object skel_ = e.m_main_ns["skel"];
    PyArrayObject *skel = reinterpret_cast<PyArrayObject*>(skel_.ptr());

    boost::python::object parents_ = e.m_main_ns["skel_parents"];
    PyArrayObject *parents = reinterpret_cast<PyArrayObject*>(parents_.ptr());

    static long frame = 0;

    const int dims = PyArray_NDIM(skel);
    if (dims > 3)
    {
      qDebug() << "Error in " << __func__ << " : too many dimensions (" << dims << " > 3)";
      return;
    }

    const long dimi = PyArray_DIM(skel, 0),
          dimj = PyArray_DIM(skel, 1),
          dimk = PyArray_DIM(skel, 2);

    if (dimk != 3)
    {
      qDebug() << "Error in " << __func__ << " : wrong 3rd dimension (" << dimk << " != 3)";
      return;
    }

    if (dimj > 2*(signed long)skel_vertices.size())
    {
      qDebug() << "Error in " << __func__ <<
        " : not enough points allocated ( " << dimj << " > "
        << skel_vertices.size() << ")";
      return;
    }

    /* Premier joint = racine, on skip */
    /* i indice de joint du squelette, j indice opengl */
    for (int i = 1, j = 0; i < dimj; ++i, j += 2)
    {
      float r = (float)i/dimj;
      int parent = *((int*) PyArray_GETPTR1(parents, i));
      double *linei = reinterpret_cast<double*>(PyArray_GETPTR2(skel, frame, i));
      double *linep = reinterpret_cast<double*>(PyArray_GETPTR2(skel, frame, parent));
      skel_vertices[j].setPosition({(float)linei[0], (float)linei[1], (float)linei[2]});
      skel_vertices[j].setColor({r, 1, 1});
      skel_vertices[j+1].setPosition({(float)linep[0], (float)linep[1], (float)linep[2]});
      skel_vertices[j+1].setColor({r, 1, 1});
    }

    m_skel_buf.bind();
    m_skel_buf.write(0, skel_vertices.data(), skel_vertices.size()*sizeof(Vertex));
    m_skel_buf.release();
    frame += 10;
    if (frame > dimi)
      frame = 0;
  }
}
