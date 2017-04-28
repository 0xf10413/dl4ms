#include "fpyeditor.h"
#include <boost/python.hpp>
#include <QDebug>

#include <sstream>
#include <tuple>

#include "input.h"

using namespace boost::python;
using namespace std;

FPyEditor::FPyEditor(FOutputScroll *output, QWidget *parent) :
  QPlainTextEdit(parent),
  m_output(output)
{
  Py_Initialize();
  m_main_module = import("__main__");
  m_main_ns = m_main_module.attr("__dict__");
}

/* Tiré de
 * http://stackoverflow.com/questions/1418015/how-to-get-python-exception-text */
std::string FPyEditor::handle_pyerror()
{
    PyObject *exc,*val,*tb;
    object formatted_list, formatted;
    PyErr_Fetch(&exc,&val,&tb);
    PyErr_NormalizeException(&exc, &val, &tb);
    handle<> hexc(exc),hval(allow_null(val)),htb(allow_null(tb));
    object traceback(import("traceback"));
    if (!tb) {
        object format_exception_only(traceback.attr("format_exception_only"));
        formatted_list = format_exception_only(hexc,hval);
    } else {
        object format_exception(traceback.attr("format_exception"));
        formatted_list = format_exception(hexc,hval,htb);
    }
    formatted = str("\n").join(formatted_list);
    return extract<std::string>(formatted);
}

void FPyEditor::keyPressEvent(QKeyEvent *e)
{
  if ((Qt::Key(e->key()) == Qt::Key_Return) &&
    (e->modifiers() & Qt::ControlModifier))
    launchPython(toPlainText());
  else
    QPlainTextEdit::keyPressEvent(e);
}

void FPyEditor::launchPython(const QString &code)
{
  /* Setup de la capture de l'output */
  const char *capture_output =
  "import sys\n"
  "class CatchOutErr:\n"
  "    def __init__(self):\n"
  "        self.value = ''\n"
  "    def write(self, txt):\n"
  "        self.value += txt\n"
  "    def __str__(self):\n"
  "        return self.value\n"
  "catchOutErr = CatchOutErr()\n"
  "sys.stdout = catchOutErr\n"
  "sys.stderr = catchOutErr\n";

  try
  {
    exec(capture_output, m_main_ns);
    exec(code.toStdString().c_str(), m_main_ns);
    string output = extract<string>(str(m_main_ns["catchOutErr"]));
    if (output.empty())
      m_output->addLine("(ok)");
    else
      m_output->addLine(QString::fromStdString(output));
  }
  catch (boost::python::error_already_set &)
  {
    try
    {
      if (PyErr_Occurred())
        m_output->addLine(QString::fromStdString(handle_pyerror()));
      else
        m_output->addLine("? There was some error of some kind");
      handle_exception();
      PyErr_Clear();
    }
    catch (boost::python::error_already_set &)
    {
      m_output->addLine("I'm so sorry,"
          "I couldn't even fetch the error.");
    }
  }
}

