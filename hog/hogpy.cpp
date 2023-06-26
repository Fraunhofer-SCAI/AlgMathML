#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>

#include "HoG.cpp"

extern "C" {
static PyObject *hogpy_hog(PyObject *self, PyObject *args) {
  PyArrayObject *image;
  unsigned long nb_bins;
  double cwidth;
  unsigned long block_size;
  PyObject* unsigned_dirs_obj;
  double clip_val;
  if (!PyArg_ParseTuple(args, "O!kdkOd", &PyArray_Type, &image, &nb_bins,
                        &cwidth, &block_size, &unsigned_dirs_obj, &clip_val)) {
    return nullptr;
  }
  const bool unsigned_dirs = PyObject_IsTrue(unsigned_dirs_obj);

  // your code goes here
  // SHEET REMOVE BEGIN
#ifndef NDEBUG
  std::cout << "nb_bins = " << nb_bins << '\n'
            << "cwidth = " << cwidth << '\n'
            << "block_size = " << block_size << '\n'
            << "unsigned_dirs = " << unsigned_dirs << '\n'
            << "clip_val = " << clip_val << '\n';
#endif

  if (!PyArray_ISFLOAT(image) || !PyArray_ISALIGNED(image) ||
      !PyArray_ISNOTSWAPPED(image) || !PyArray_IS_F_CONTIGUOUS(image)) {
    PyErr_SetString(PyExc_ValueError, "The image must contain float values, must be aligned, not swapped and in F storage order");
    return nullptr;
  }

  const size_t ndim = PyArray_NDIM(image);
  if (ndim != 3 && ndim != 2) {
    PyErr_SetString(PyExc_ValueError, "The image must have 2 or 3 dimensions");
    return nullptr;
  }

  const npy_intp *dims = PyArray_DIMS(image);
  size_t img_size[2] = {static_cast<size_t>(dims[0]), static_cast<size_t>(dims[1])};
  if (ndim == 3 && dims[2] != 3) {
    PyErr_SetString(PyExc_ValueError, "The image must have 3 color channels in the last dimension");
    return nullptr;
  }


  const npy_intp *strides = PyArray_STRIDES(image);
  const double *pixels = static_cast<double *>(PyArray_DATA(image));
  const size_t stride = strides[1] / sizeof(*pixels);
  const size_t channel_stride = ndim == 3 ? strides[2] / sizeof(*pixels) : 0;

  long out_len =
      static_cast<long>(getNumFeatures(img_size, nb_bins, cwidth, block_size));
  PyObject *out_py = PyArray_SimpleNew(1, &out_len, NPY_DOUBLE);
  if (out_py == nullptr) {
    return nullptr;
  }
  double *out = static_cast<double *>(
      PyArray_DATA(reinterpret_cast<PyArrayObject *>(out_py)));
  HoG(pixels, nb_bins, cwidth, block_size, unsigned_dirs, clip_val, img_size,
      stride, out, ndim == 2, channel_stride);

  return out_py;
  // SHEET REMOVE END
}

static PyMethodDef HogpyMethods[] = {
    {"hog", hogpy_hog, METH_VARARGS,
     "Compute the HOG feature vector for an image."},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef hogymodule = {
    PyModuleDef_HEAD_INIT, "hogpy", /* name of module */
    nullptr,                        /* module documentation, may be nullptr */
    -1, /* size of per-interpreter state of the module,
           or -1 if the module keeps state in global variables. */
    HogpyMethods,
    nullptr, /* slots */
    nullptr, /* traverse */
    nullptr, /* clear */
    nullptr, /* free*/
};
}

PyMODINIT_FUNC PyInit_hogpy() {
  import_array();
  return PyModule_Create(&hogymodule);
}
