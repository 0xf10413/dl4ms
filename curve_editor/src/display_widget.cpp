/*
 * display_widget.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "display_widget.h"
#include <QDebug>
#include <QString>
#include <QOpenGLShaderProgram>
#include "vertex.hpp"

static const Vertex sg_vertexes[] = {
  Vertex( QVector3D( 0.00f,  0.75f, 1.0f), QVector3D(1.0f, 0.0f, 0.0f) ),
  Vertex( QVector3D( 0.75f, -0.75f, 1.0f), QVector3D(0.0f, 1.0f, 0.0f) ),
  Vertex( QVector3D(-0.75f, -0.75f, 1.0f), QVector3D(0.0f, 0.0f, 1.0f) )
};

void DisplayWidget::initializeGL()
{
  // Initialize OpenGL Backend
  initializeOpenGLFunctions();
  printVersionInformation();

  // Set global information
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  // Application-specific initialization
  {
    // Create Shader (Do not release until VAO is created)
    m_program = new QOpenGLShaderProgram();
    m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/simple.vert");
    m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/simple.frag");
    m_program->link();
    m_program->bind();

    // Create Buffer (Do not release until VAO is created)
    m_vertex.create();
    m_vertex.bind();
    m_vertex.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vertex.allocate(sg_vertexes, sizeof(sg_vertexes));

    // Create Vertex Array Object
    m_object.create();
    m_object.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, Vertex::positionOffset(), Vertex::PositionTupleSize, Vertex::stride());
    m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());

    // Release (unbind) all
    m_object.release();
    m_vertex.release();
    m_program->release();
  }
}

DisplayWidget::DisplayWidget(QWidget *parent) : QOpenGLWidget(parent)
{
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  setFormat(format);

  resize(500,500);
  setMinimumSize(QSize(400,200));
}

void DisplayWidget::paintGL()
{
 // Clear
  glClear(GL_COLOR_BUFFER_BIT);

  // Render using our shader
  m_program->bind();
  {
    m_object.bind();
    glDrawArrays(GL_TRIANGLES, 0,
        sizeof(sg_vertexes) / sizeof(sg_vertexes[0]));
    m_object.release();
  }
  m_program->release();
}

void DisplayWidget::teardownGL()
{
  // Actually destroy our OpenGL information
  m_object.destroy();
  m_vertex.destroy();
  delete m_program;
}


void DisplayWidget::printVersionInformation()
{
  qDebug() << "Warning : printVersionInformation not implemented";
}
