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

constexpr size_t NB_FRAMES = 7200;
constexpr size_t NB_JOINS = 22;

using std::array;

static array<Vertex,NB_FRAMES> curve_vertices { };
static std::vector<array<Vertex,2*NB_JOINS>> skel_vertices {NB_FRAMES};

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
  m_current_frame = 0;
  m_fps = 1;

  m_transform.translate(.0, -1.0, -5.0f);
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  setFormat(format);

  setMinimumSize(QSize(400,400));
  setMaximumSize(QSize(400,400));
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
    glDrawArrays(GL_LINES, 0, 2*NB_JOINS*sizeof(Vertex));
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

  static const float transSpeed = 0.5f;
  QVector3D translation;

  // Camera Transformation
  if (Input::buttonPressed(Qt::LeftButton))
  {
    setFocus();
    static const float rotSpeed   = 0.5f;

    // Handle rotations
    m_camera.rotate(-rotSpeed * Input::mouseDelta().x(), Camera3D::LocalUp);
    m_camera.rotate(-rotSpeed * Input::mouseDelta().y(), m_camera.right());

  }

  // Handle translations
  if (Input::keyPressed(Qt::Key_Z))
    translation += m_camera.forward();
  if(Input::buttonPressed(Qt::MidButton))
    translation += m_camera.forward();

  if (Input::keyPressed(Qt::Key_S))
    translation -= m_camera.forward();
  if (Input::buttonPressed(Qt::RightButton))
    translation -= m_camera.forward();

  if (Input::keyPressed(Qt::Key_Q))
    translation -= m_camera.right();

  if (Input::keyPressed(Qt::Key_D))
    translation += m_camera.right();

  if (Input::keyPressed(Qt::Key_Shift))
    translation -= m_camera.up();

  if (Input::keyPressed(Qt::Key_Space))
    translation += m_camera.up();

  m_camera.translate(transSpeed * translation);

  // Update instance information
  //m_transform.rotate(1.0f, QVector3D(0.0f, 1.0f, 0.0f));
  m_current_frame += m_fps;
  if (m_current_frame >= (signed int) NB_FRAMES)
    m_current_frame = 0;
  if (m_current_frame < 0)
    m_current_frame = NB_FRAMES - 1;

  m_skel_buf.bind();
  m_skel_buf.write(0, skel_vertices[m_current_frame].data(),
      NB_JOINS*2*sizeof(Vertex));
  m_skel_buf.release();

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

    std::array<float,3> cur_pos {0.,0.,0.};
    curve_vertices[0].setPosition({0.f, 0.f, 0.f});
    curve_vertices[0].setColor({1, 1, 0});
    float prev_theta = 0.;
    for (int j = 1; j < dimj; ++j)
    {
      float b = (float)j/dimj;
      float vx = *(float*)PyArray_GETPTR2(curve, 0, j);
      float vy = *(float*)PyArray_GETPTR2(curve, 1, j);

      cur_pos[0] += cos(prev_theta)*vx - sin(prev_theta)*vy;
      cur_pos[1] += sin(prev_theta)*vx + cos(prev_theta)*vy;
      curve_vertices[j].setPosition({cur_pos[0], 0.f, cur_pos[1]});
      curve_vertices[j].setColor({1, 1, b});
      prev_theta += *(float*)PyArray_GETPTR2(curve, 2, j);
    }
    for (size_t j = dimj; j < curve_vertices.size(); ++j)
    {
      curve_vertices[j].setPosition({(cur_pos[0]),
          (cur_pos[1]),
          (cur_pos[2])});
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

    const int dims = PyArray_NDIM(skel);
    if (dims > 3)
    {
      qDebug() << "Error in " << __func__ << " : too many dimensions (" << dims << " > 3)";
      return;
    }

    const long dimi = PyArray_DIM(skel, 0),
          dimj = PyArray_DIM(skel, 1),
          dimk = PyArray_DIM(skel, 2);

    if (dimi != NB_FRAMES)
    {
      qDebug() << "Error in " << __func__ <<
        " : wrong number of frames ( " << dimi << " != "
        << NB_FRAMES << ")";
      return;
    }


    if (dimj > 2*(signed long)skel_vertices.size())
    {
      qDebug() << "Error in " << __func__ <<
        " : not enough points allocated ( " << dimj << " > "
        << 2*skel_vertices.size() << ")";
      return;
    }

    if (dimk != 3)
    {
      qDebug() << "Error in " << __func__ << " : wrong 3rd dimension (" << dimk << " != 3)";
      return;
    }

    for (int i = 0; i < dimi; ++i)
    {
      for (int j = 0; j < dimj; ++j)
      {
        float r = (float)j/dimj;
        int parent = *((int*) PyArray_GETPTR1(parents, j));

        float x = *(float*)PyArray_GETPTR3(skel, i, j, 0);
        float y = *(float*)PyArray_GETPTR3(skel, i, j, 1);
        float z = *(float*)PyArray_GETPTR3(skel, i, j, 2);
        skel_vertices[i][2*j].setPosition({x, y, z});
        skel_vertices[i][2*j].setColor({r, 1, 1});

        if (parent != -1)
        {
          float px = *(float*)PyArray_GETPTR3(skel, i, parent, 0);
          float py = *(float*)PyArray_GETPTR3(skel, i, parent, 1);
          float pz = *(float*)PyArray_GETPTR3(skel, i, parent, 2);
          skel_vertices[i][2*j+1].setPosition({px, py, pz});
          skel_vertices[i][2*j+1].setColor({r, 1, 1});
        }
        else // Racine
        {
          skel_vertices[i][2*j+1].setPosition({x, y, z});
          skel_vertices[i][2*j+1].setColor({r, 1, 1});
        }
      }
    }
  }
}

void DisplayWidget::play_pause()
{
  m_fps = !m_fps;
}

void DisplayWidget::stop()
{
  m_fps = 0;
  m_current_frame = 0;
}

void DisplayWidget::faster()
{
  ++m_fps;
  if (m_fps == 0)
    ++m_fps;
}

void DisplayWidget::slower()
{
  --m_fps;
  if (m_fps == 0)
    --m_fps;
}
