#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup script for GDAL Python bindings.
# Inspired by psycopg2 setup.py file
# http://www.initd.org/tracker/psycopg/browser/psycopg2/trunk/setup.py
# Howard Butler hobu.inc@gmail.com


import os
import sys
from glob import glob

from setuptools.command.build_ext import build_ext
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension

version = '3.8.4'

# If CXX is defined in the environment, it will be used to link the .so
# but setuptools will be confused if it is made of several words like 'ccache g++'
# and it will try to use only the first word.
# See https://lists.osgeo.org/pipermail/gdal-dev/2016-July/044686.html
# Note: in general when doing "make", CXX will not be defined, unless it is defined as
# an environment variable, but in that case it is the value of GDALmake.opt that
# will be set, not the one from the environment that started "make" !
# If no CXX environment variable is defined, then the value of the CXX variable
# in GDALmake.opt will not be set as an environment variable
if 'CXX' in os.environ and os.environ['CXX'].strip().find(' ') >= 0:
    if os.environ['CXX'].strip().startswith('ccache ') and os.environ['CXX'].strip()[len('ccache '):].find(' ') < 0:
        os.environ['CXX'] = os.environ['CXX'].strip()[len('ccache '):]
    else:
        print('WARNING: "CXX=%s" was defined in the environment and contains more than one word. Unsetting it since that is incompatible of setuptools' % os.environ['CXX'])
        del os.environ['CXX']
if 'CC' in os.environ and os.environ['CC'].strip().find(' ') >= 0:
    if os.environ['CC'].strip().startswith('ccache ') and os.environ['CC'].strip()[len('ccache '):].find(' ') < 0:
        os.environ['CC'] = os.environ['CC'].strip()[len('ccache '):]
    else:
        print('WARNING: "CC=%s" was defined in the environment and contains more than one word. Unsetting it since that is incompatible of setuptools' % os.environ['CC'])
        del os.environ['CC']

# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

is_standalone_build = not os.path.exists('/home/even/gdal/3.8/build/swig/python')

if is_standalone_build:
    include_dirs = []
    library_dirs = []
else:
    include_dirs = ['/home/even/gdal/3.8/build/port', '/home/even/gdal/3.8/port', '/home/even/gdal/3.8/build/gcore', '/home/even/gdal/3.8/gcore', '/home/even/gdal/3.8/alg', '/home/even/gdal/3.8/ogr/', '/home/even/gdal/3.8/ogr/ogrsf_frmts', '/home/even/gdal/3.8/gnm', '/home/even/gdal/3.8/apps']
    library_dirs = ['/home/even/gdal/3.8/build']
libraries = ['gdal']


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

# Function to find numpy's include directory
def get_numpy_include():
    # Fix numpy installation using setuptools
    # __builtins__ can be either a dict or a module
    def _set_builtin(name, value):
        if isinstance(__builtins__, dict):
            __builtins__[name] = value
        else:
            setattr(__builtins__, name, value)

    _set_builtin("__NUMPY_SETUP__", False)

    import numpy
    return numpy.get_include()


class gdal_config_error(Exception):
    pass


def fetch_config(option, gdal_config='gdal-config'):

    command = gdal_config + " --%s" % option

    import subprocess
    command, args = command.split()[0], command.split()[1]
    try:
        p = subprocess.Popen([command, args], stdout=subprocess.PIPE)
    except OSError:
        e = sys.exc_info()[1]
        raise gdal_config_error(e)
    r = p.stdout.readline().decode('ascii').strip()
    p.stdout.close()
    p.wait()

    return r


def supports_cxx11(compiler, compiler_flag=None):
    ret = False
    with open('gdal_python_cxx11_test.cpp', 'wt') as f:
        f.write("""
#if __cplusplus < 201103L
#error "C++11 required"
#endif
int main () { return 0; }""")
        f.close()
        extra_postargs = None
        if compiler_flag:
            extra_postargs = [compiler_flag]

        if os.name == 'posix':
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            devnull = open(os.devnull, 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            try:
                compiler.compile([f.name], extra_postargs=extra_postargs)
                ret = True
            except Exception:
                pass
            os.dup2(oldstderr, sys.stderr.fileno())
            devnull.close()
        else:
            try:
                compiler.compile([f.name], extra_postargs=extra_postargs)
                ret = True
            except Exception:
                pass
    os.unlink('gdal_python_cxx11_test.cpp')
    if os.path.exists('gdal_python_cxx11_test.o'):
        os.unlink('gdal_python_cxx11_test.o')
    return ret

###Based on: https://stackoverflow.com/questions/28641408/how-to-tell-which-compiler-will-be-invoked-for-a-python-c-extension-in-setuptool
def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except Exception:
            return False
    return True

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

numpy_include_dir = '.'
try:
    numpy_include_dir = get_numpy_include()
    HAVE_NUMPY = numpy_include_dir != '.'
    if not HAVE_NUMPY:
        print("WARNING: numpy found, but numpy headers were not found!  Array support will not be enabled")
except ImportError:
    HAVE_NUMPY = False
    print('WARNING: numpy not available!  Array support will not be enabled')


class gdal_ext(build_ext):

    GDAL_CONFIG = 'gdal-config'
    user_options = build_ext.user_options[:]
    user_options.extend([
        ('gdal-config=', None,
         "The name of the gdal-config binary and/or a full path to it"),
    ])

    def run(self):
        build_ext.run(self)

    def initialize_options(self):
        global numpy_include_dir
        build_ext.initialize_options(self)

        self.numpy_include_dir = numpy_include_dir
        self.gdal_config = self.GDAL_CONFIG
        self.extra_cflags = []
        self.parallel = True

    def get_compiler(self):
        return self.compiler or ('msvc' if os.name == 'nt' else 'unix')


    def get_gdal_config(self, option):
        try:
            return fetch_config(option, gdal_config=self.gdal_config)
        except gdal_config_error:
            msg = 'Could not find gdal-config. Make sure you have installed the GDAL native library and development headers.'
            import sys
            import traceback
            traceback_string = ''.join(traceback.format_exception(*sys.exc_info()))
            raise gdal_config_error(traceback_string + '\n' + msg)


    def build_extensions(self):

        # Add a -std=c++11 or similar flag if needed
        ct = self.compiler.compiler_type
        if ct == 'unix' and not supports_cxx11(self.compiler):
            cxx11_flag = None
            if supports_cxx11(self.compiler, '-std=c++11'):
                cxx11_flag = '-std=c++11'
            if cxx11_flag:
                for ext in self.extensions:
                    # gdalconst builds as a .c file
                    if ext.name != 'osgeo._gdalconst':
                        ext.extra_compile_args += [cxx11_flag]

                    # Adding arch flags here if OS X and compiler is clang
                    if sys.platform == 'darwin' and [int(x) for x in os.uname()[2].split('.')] >= [11, 0, 0]:
                        # since MacOS X 10.9, clang no longer accepts -mno-fused-madd
                        # extra_compile_args.append('-Qunused-arguments')
                        clang_flag = '-Wno-error=unused-command-line-argument-hard-error-in-future'
                        if has_flag(self.compiler, clang_flag):
                            ext.extra_compile_args += [clang_flag]
                        else:
                            clang_flag = '-Wno-error=unused-command-line-argument'
                            if has_flag(self.compiler, clang_flag):
                                ext.extra_compile_args += [clang_flag]

        build_ext.build_extensions(self)

    def finalize_options(self):
        global include_dirs, library_dirs

        try:
            library_version = self.get_gdal_config("version").replace("dev", "")
            library_version_num = [int(x) for x in library_version.split('.')]
            gdal_python_version = [int(x) for x in version.split('.')]
            if library_version_num < gdal_python_version:
                raise Exception(f"Python bindings of GDAL {version} require at least libgdal {version}, but {library_version} was found")
        except gdal_config_error:
            pass

        include_dirs_found = self.include_dirs is not None or len(include_dirs) != 0

        if self.include_dirs is None:
            self.include_dirs = include_dirs
        # Needed on recent MacOSX
        elif include_dirs and isinstance(self.include_dirs, str) and sys.platform == 'darwin':
            self.include_dirs += ':' + ':'.join(include_dirs)

        if self.library_dirs is None:
            self.library_dirs = library_dirs
        # Needed on recent MacOSX
        elif library_dirs and isinstance(self.library_dirs, str) and sys.platform == 'darwin':
            self.library_dirs += ':' + ':'.join(library_dirs)

        if self.libraries is None:
            self.libraries = libraries

        build_ext.finalize_options(self)

        if numpy_include_dir != '.':
	        self.include_dirs.append(self.numpy_include_dir)

        if self.get_compiler() == 'msvc':
            return

        if not include_dirs_found:
            # Get paths from gdal-config
            gdaldir = self.get_gdal_config('prefix')
            self.library_dirs.append(os.path.join(gdaldir, 'lib'))
            self.include_dirs.append(os.path.join(gdaldir, 'include'))

            cflags = self.get_gdal_config('cflags')
            if cflags:
                self.extra_cflags = cflags.split()

    def build_extension(self, ext):
        # We override this instead of setting extra_compile_args directly on
        # the Extension() instantiations below because we want to use the same
        # logic to resolve the location of gdal-config throughout.
        ext.extra_compile_args.extend(self.extra_cflags)
        return build_ext.build_extension(self, ext)


extra_link_args = []
extra_compile_args = []

gdal_module = Extension('osgeo._gdal',
                        sources=['extensions/gdal_wrap.cpp'],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args)

gdalconst_module = Extension('osgeo._gdalconst',
                             sources=['extensions/gdalconst_wrap.c'],
                             extra_compile_args=extra_compile_args,
                             extra_link_args=extra_link_args)

osr_module = Extension('osgeo._osr',
                       sources=['extensions/osr_wrap.cpp'],
                       extra_compile_args=extra_compile_args,
                       extra_link_args=extra_link_args)

ogr_module = Extension('osgeo._ogr',
                       sources=['extensions/ogr_wrap.cpp'],
                       extra_compile_args=extra_compile_args,
                       extra_link_args=extra_link_args)


array_module = Extension('osgeo._gdal_array',
                         sources=['extensions/gdal_array_wrap.cpp'],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args)

gnm_module = Extension('osgeo._gnm',
                       sources=['extensions/gnm_wrap.cpp'],
                       extra_compile_args=extra_compile_args,
                       extra_link_args=extra_link_args)

ext_modules = [gdal_module,
               gdalconst_module,
               osr_module,
               ogr_module]

GNM_ENABLED = True
if GNM_ENABLED:
    ext_modules.append(gnm_module)

if HAVE_NUMPY:
    ext_modules.append(array_module)

utils_package_root = 'gdal-utils'   # path for gdal-utils sources
packages = find_packages(utils_package_root)
packages = ['osgeo'] + packages
package_dir = {'osgeo': 'osgeo', '': utils_package_root}

readme = open('README.rst', encoding="utf-8").read()

name = 'GDAL'
author = "Frank Warmerdam"
author_email = "warmerdam@pobox.com"
maintainer = "Howard Butler"
maintainer_email = "hobu.inc@gmail.com"
description = "GDAL: Geospatial Data Abstraction Library"
license_type = "MIT"
url = "http://www.gdal.org"

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: C',
    'Programming Language :: C++',
    'Topic :: Scientific/Engineering :: GIS',
    'Topic :: Scientific/Engineering :: Information Analysis',

]

exclude_package_data = {'': ['CMakeLists.txt']}

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    long_description=readme,
    long_description_content_type='text/x-rst',
    description=description,
    license=license_type,
    classifiers=classifiers,
    packages=packages,
    package_dir=package_dir,
    url=url,
    python_requires='>=3.6.0',
    ext_modules=ext_modules,
    scripts=glob(utils_package_root + '/scripts/*.py'),
    # This must *not* be conditionalized by HAVE_NUMPY, since this is for a "pip install gdal[numpy]" type of installation
    extras_require={'numpy': ['numpy > 1.0.0']},
    zip_safe=False,
    cmdclass=dict(build_ext=gdal_ext),
    exclude_package_data = exclude_package_data,
)
