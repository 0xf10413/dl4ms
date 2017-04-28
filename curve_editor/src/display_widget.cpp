/*
 * display_widget.cpp
 * Copyright (C) 2017 flodeu <flodeu@W8Debian>
 *
 * Distributed under terms of the MIT license.
 */

#include "display_widget.h"
#include <QDebug>
#include <QString>
#include <QKeyEvent>
#include <QOpenGLShaderProgram>
#include "vertex.hpp"
#include "input.h"

// Front Verticies
#define VERTEX_FIRST Vertex( QVector3D( 0.f,  1.0f,  0.f), QVector3D( 1.0f, 0.0f, 0.0f ) )
#define VERTEX_SECOND Vertex( QVector3D( 0.5f,  0.f,  0.f), QVector3D( .0f, 1.0f, 0.0f ) )
#define VERTEX_THIRD Vertex( QVector3D( 0.f,  0.f,  -0.5f), QVector3D( 0.0f, 0.0f, 1.0f ) )
#define VERTEX_FOURTH Vertex( QVector3D( 0.f,  0.f, 0.5f), QVector3D( 1.0f, 1.0f, 0.0f ) )

// Create a colored pyramide

static const Vertex sg_vertexes[] = {
  VERTEX_FIRST, VERTEX_SECOND, VERTEX_THIRD,
  VERTEX_FIRST, VERTEX_THIRD, VERTEX_FOURTH,
  VERTEX_FIRST, VERTEX_FOURTH, VERTEX_SECOND,
  VERTEX_SECOND, VERTEX_THIRD, VERTEX_FOURTH,
};

#undef VERTEX_BBR
#undef VERTEX_BBL
#undef VERTEX_BTL
#undef VERTEX_BTR

#undef VERTEX_FBR
#undef VERTEX_FBL
#undef VERTEX_FTL
#undef VERTEX_FTR

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
	m_transform.translate(.0, .0, -5.0f);
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	setFormat(format);

	setMinimumSize(QSize(400,200));
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
		static const float transSpeed = 0.5f;
		static const float rotSpeed   = 0.5f;

		// Handle rotations
		m_camera.rotate(-rotSpeed * Input::mouseDelta().x(), Camera3D::LocalUp);
		m_camera.rotate(-rotSpeed * Input::mouseDelta().y(), m_camera.right());

		// Handle translations
		QVector3D translation;
		if (Input::keyPressed(Qt::Key_Z))
		{
			translation += m_camera.forward();
		}
		if (Input::keyPressed(Qt::Key_S))
		{
			translation -= m_camera.forward();
		}
		if (Input::keyPressed(Qt::Key_Q))
		{
			translation -= m_camera.right();
		}
		if (Input::keyPressed(Qt::Key_D))
		{
			translation += m_camera.right();
		}

		if (Input::keyPressed(Qt::Key_A))
		{
			translation -= m_camera.up();
		}
		if (Input::keyPressed(Qt::Key_E))
		{
			translation += m_camera.up();
		}
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
