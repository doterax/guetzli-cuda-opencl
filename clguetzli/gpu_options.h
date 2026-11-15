/*
 * Options for GPU backend initialization, passed in from the configuration
 * (app/application.cpp) instead of global variables.
 */
#pragma once

struct GpuOptions {
  int device = -1;       // device index within the backend; < 0 = default selection
  bool verbose = false;  // print device details on initialization (--verbose)
};
