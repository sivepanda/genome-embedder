#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "gcc.h"

static PyObject* coverage_vectors_from_str(PyObject *self, PyObject *args) {
    const char* inputString;

    if (!PyArg_ParseTuple(args, "s", &inputString)) {
        return NULL;
    }
    
    import_array(); // ini Numpy array API
    

    int lines = 0;
    for (const char *p = inputString; *p; p++) {
        if(*p == '\n') lines++;
    }

    npy_float16* data = malloc(lines * sizeof(npy_float16));
    if (data == NULL) return PyErr_NoMemory();

    const char *line_start = input_string;
    const char *line_end;
    int i = 0;

    while (*line_start) {
        line_end = strchr(line_start, '\n') ? strchr(line_start, '\n') : line_start + strlen(line_start);
        const char *last_tab = strrchr(line_start, '\t');

        if (last_tab && last_tab < line_end) {
            float value = atof(last_tab + 1);
            data[i++] = value;  // Cast to float16 implicitly, may need explicit conversion depending on platform
        }

        if (*line_end == '\n') {
            line_start = line_end + 1;
        } else {
            break;  // No more lines
        }
    }

    // Create NumPy array
    npy_intp dims[1] = {lines};  // Array dimension
    PyObject* numpyArray = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT16, data);
    PyArray_ENABLEFLAGS((PyArrayObject*)numpyArray, NPY_ARRAY_OWNDATA);

    return numpyArray;
}

static PyObject* dda(PyObject *self, PyObject *args) {
    int a, b;

    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }

    int sum = a + b;

    return Py_BuildValue("i", sum);
}

static PyModuleDef ModuleMethods[] = {
    {"dda", dda, METH_VARARGS, "Adds two integers"},
    {"coverage_vectors_from_string", coverage_vectors_from_string, METH_VARARGS, "Converts String generated from Bedtools to Numpy array"},
    {NULL, NULL, 0, NULL} //Sentinel?
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "generate_coverage_vectors",  // Module name
    "Module documentation",
    -1,
    ModuleMethods
};

PyMODINIT_FUNC PyInit_generate_coverage_vectors(void) {
    return PyModule_Create(&moduledef);
}
