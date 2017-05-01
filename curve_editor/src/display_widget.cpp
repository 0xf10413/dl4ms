/*
 * display_widget.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "display_widget.h"
#include "vertex.hpp"
#include "input.h"

#include <QDebug>
#include <QString>
#include <QKeyEvent>
#include <QOpenGLShaderProgram>

#include <numpy/arrayobject.h>

#include <array>
#include <random>

static std::array<Vertex,600> sg_vertexes { };

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

		// Create Buffer (Do not release until VAO is created)
		m_vertex.create();
		m_vertex.bind();
		m_vertex.setUsagePattern(QOpenGLBuffer::DynamicDraw);
		m_vertex.allocate(sg_vertexes.data(), sg_vertexes.size());

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
	m_transform.translate(.0, .0, -5.0f);
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	setFormat(format);

	setMinimumSize(QSize(400,200));
  sg_vertexes[0] = Vertex(QVector3D{1,1,1}, QVector3D{1,0,0} );
  sg_vertexes[1] = Vertex(QVector3D{1,0,0}, QVector3D{0,1,0} );
  sg_vertexes[2] = Vertex(QVector3D{1,0,0}, QVector3D{0,1,0} );
  sg_vertexes[3] = Vertex(QVector3D{0,1,0}, QVector3D{0,0,1} );
  sg_vertexes[4] = Vertex(QVector3D{0,0,1}, QVector3D{1,0,1} );

  std::uniform_real_distribution<float> unif(0,1);
  std::mt19937 mersenne(41);

  /* Random colors */
  for (auto &vertex : sg_vertexes)
    vertex.setColor({unif(mersenne), unif(mersenne), unif(mersenne)});
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
		m_object.bind();
		m_program->setUniformValue(u_modelToWorld, m_transform.toMatrix());
		glDrawArrays(GL_LINES, 0, sg_vertexes.size());
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
	m_transform.rotate(1.0f, QVector3D(0.4f, 0.3f, 0.3f));

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

void DisplayWidget::refreshDataToPrint(PyArrayObject *ptr)
{
  qDebug() << __func__ << " called";
  const int dims = PyArray_NDIM(ptr);
  if (dims > 2)
  {
    qDebug() << "Error in " << __func__ << " : too many dimensions (" << dims << " > 2)";
    return;
  }

  const long dimi = PyArray_DIM(ptr, 0), dimj = PyArray_DIM(ptr, 1);
  if (dimj != 3)
  {
    qDebug() << "Error in " << __func__ << " : wrong 2nd dimension (" << dimj << " != 3)";
    return;
  }

  if (dimi > 2*(signed long)sg_vertexes.size())
  {
    qDebug() << "Error in " << __func__ <<
      " : not enough points allocated ( " << dimi << " > "
      << sg_vertexes.size() << ")";
  }

  for (int i = 0, j = 0; i < dimi-1; ++i, j += 2)
  {
    double *linei = reinterpret_cast<double*>(PyArray_GETPTR1(ptr, i));
    double *linej = reinterpret_cast<double*>(PyArray_GETPTR1(ptr, i+1));
    sg_vertexes[j].setPosition({(float)linei[0], (float)linei[1], (float)linei[2]});
    sg_vertexes[j+1].setPosition({(float)linej[0], (float)linej[1], (float)linej[2]});
  }
  m_vertex.bind();
  m_vertex.write(0, sg_vertexes.data(), sg_vertexes.size());
  m_vertex.release();
}
