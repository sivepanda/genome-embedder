#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "gcc.h"

char* split_first_line(char **str) {
    if (str == NULL || *str == NULL) {
        return NULL;
    }

    char *newline = strchr(*str, '\n');
    char *firstLine = NULL;

    if (newline != NULL) {
        size_t len = newline - *str;
        firstLine = malloc(len + 1);
        if (firstLine == NULL) {
            return NULL;
        }
        strncpy(firstLine, *str, len);
        firstLine[len] = '\0';
        *str = newline + 1;
    } else {
        firstLine = malloc(strlen(*str) + 1);
        if (firstLine == NULL) {
            return NULL;
        }
        strcpy(firstLine, *str);
        *str = "";  // or *str = NULL if you prefer to clear the pointer
    }

    return firstLine;
}


static PyObject* coverage_vectors_from_string(PyObject *self, PyObject *args) {
    const char* inputString;
    char* inString;

    if (!PyArg_ParseTuple(args, "s", &inputString)) {
        return NULL;
    }

    inString = malloc(strlen(inputString) + 1);
    if(!inString) {
        PyErr_NoMemory();
        return NULL;
    }

    strcpy(inString, inputString);
    
    import_array(); // ini Numpy array API
    

    printf("\nINPUT-----%s", inputString);
    int lines = 1;
    for (const char *p = inputString; *p; p++) {
        if(*p == '\n') lines++;
        printf("\n%i", lines);

    }

    npy_float16* data = malloc(lines * sizeof(npy_float16));
    if (data == NULL) return PyErr_NoMemory();
    int i = 0;

    while (*inString) {
        printf("\nhere0");
        char *firstLine = split_first_line(&inString);

        const char *last_tab = strrchr(firstLine, '\t');
        printf("\n firstLine %s", firstLine);
        printf("\n here1");
        printf("\n lnend %s", last_tab);
        printf("\n atof %f", atof(last_tab + 1));

        if (last_tab) {
            printf("\n conv");
            float value = atof(last_tab + 1);
            printf("\n val %f", value);
            data[i++] = value;  // Cast to float16 implicitly, may need explicit conversion depending on platform
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
    //{"dda", dda, METH_VARARGS, "Adds two integers"},
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
