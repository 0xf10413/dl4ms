#include <boost/python.hpp>
#include <iostream>
#include <tuple>

using namespace boost::python;
using namespace std;

int f()
{
  Py_Initialize();

  try
  {
    object main_module = import("__main__");
    object main_ns = main_module.attr("__dict__");
    string to_exec = "print('Hello world!')\nprnt('Bye world!')";
    exec(to_exec.c_str(), main_ns);
  }
  catch (boost::python::error_already_set &)
  {
    PyObject *e, *v, *t;
    PyErr_Fetch(&e, &v, &t);

    if (!e) return 0;

    object e_obj(handle<>(allow_null(e)));
    object v_obj(handle<>(allow_null(v)));
    object t_obj(handle<>(allow_null(t)));

    string err_message = extract<string>(v_obj);
    long lineno = extract<long>(t_obj.attr("tb_lineno"));
    string filename = extract<string>(t_obj.attr("tb_frame")
        .attr("f_code").attr("co_filename"));
    string funcname = extract<string>(t_obj.attr("tb_frame")
        .attr("f_code").attr("co_name"));

    cerr << "Got python error, at line " << lineno << " : "
      << endl << "--> " << err_message << endl;
    cerr << "Backtrace : "
      << "--> File = " << filename << endl
      << "--> Function = " << funcname << endl;
    return EXIT_FAILURE;
  }

  cout << "Execution ok" << endl;
  return EXIT_SUCCESS;
}

