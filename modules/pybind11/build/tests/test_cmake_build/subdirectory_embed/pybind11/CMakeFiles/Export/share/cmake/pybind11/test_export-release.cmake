#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pybind11::test_embed_lib" for configuration "Release"
set_property(TARGET pybind11::test_embed_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pybind11::test_embed_lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libtest_embed_lib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS pybind11::test_embed_lib )
list(APPEND _IMPORT_CHECK_FILES_FOR_pybind11::test_embed_lib "${_IMPORT_PREFIX}/bin/libtest_embed_lib.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
