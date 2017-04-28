######################################################################
# Automatically generated by qmake (3.0) jeu. avr. 27 14:18:06 2017
######################################################################

TEMPLATE = app
TARGET = curve_editor
INCLUDEPATH += . /usr/include/python3.5m
LIBS += -lpython3.5m -lboost_python-py35
RESOURCES += rc/main.qrc
CONFIG += no_keywords debug c++11

DESTDIR = build
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR

launch.target =
launch.commands = $$DESTDIR/$$TARGET
launch.depends = $$DESTDIR/$$TARGET
QMAKE_EXTRA_TARGETS += launch

# Input
HEADERS += src/mainwindow.h src/display_widget.h src/vertex.hpp \
            src/fpyeditor.h src/foutput_scroll.h src/transform3d.h \
            src/camera3d.h src/input.h
SOURCES += src/main.cpp src/mainwindow.cpp src/fpyeditor.cpp \
           src/display_widget.cpp src/foutput_scroll.cpp \
           src/transform3d.cpp src/camera3d.cpp src/input.cpp
QT += widgets gui