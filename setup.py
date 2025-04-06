"""Setup."""

import os
from distutils import log
from distutils._modified import newer_group  # type: ignore  # noqa: PGH003
from distutils.errors import DistutilsSetupError

import numpy as np
from setuptools import Extension, setup
from setuptools.command import build_ext


class BuildExtMod(build_ext.build_ext):
    """Custom version of build_ext."""

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Initialize the class."""
        self._skipped_extensions = set()
        super().__init__(*args, **kwargs)

    def build_extension(self, ext: Extension) -> None:  # noqa: C901
        """Build an extension module."""
        if not self.editable_mode or self.force:
            super().build_extension(ext)
            return

        ### Cython distutils build_ext
        from Cython.Build.Dependencies import cythonize

        # Set up the include_path for the Cython compiler:
        #    1.    Start with the command line option.
        #    2.    Add in any (unique) paths from the extension
        #        cython_include_dirs (if Cython.Distutils.extension is used).
        #    3.    Add in any (unique) paths from the extension include_dirs
        includes = list(self.cython_include_dirs)
        for include_dir in getattr(ext, "cython_include_dirs", []):
            if include_dir not in includes:
                includes.append(include_dir)

        # In case extension.include_dirs is a generator, evaluate it and keep
        # result
        ext.include_dirs = list(ext.include_dirs)
        for include_dir in ext.include_dirs + list(self.include_dirs):
            if include_dir not in includes:
                includes.append(include_dir)

        # Set up Cython compiler directives:
        #    1. Start with the command line option.
        #    2. Add in any (unique) entries from the extension
        #         cython_directives (if Cython.Distutils.extension is used).
        directives = dict(self.cython_directives)
        if hasattr(ext, "cython_directives"):
            directives.update(ext.cython_directives)

        if self.get_extension_attr(ext, "cython_cplus"):
            ext.language = "c++"

        options = {
            "use_listing_file": self.get_extension_attr(ext, "cython_create_listing"),
            "emit_linenums": self.get_extension_attr(ext, "cython_line_directives"),
            "include_path": includes,
            "compiler_directives": directives,
            "build_dir": self.build_temp if self.get_extension_attr(ext, "cython_c_in_temp") else None,
            "generate_pxi": self.get_extension_attr(ext, "cython_gen_pxi"),
            "gdb_debug": self.get_extension_attr(ext, "cython_gdb"),
            "c_line_in_traceback": not getattr(ext, "no_c_in_traceback", 0),
            "compile_time_env": self.get_extension_attr(ext, "cython_compile_time_env", default=None),
        }

        new_ext = cythonize(ext, force=self.force, quiet=self.verbose == 0, **options)[0]
        ext.sources = new_ext.sources
        ext.depends = new_ext.depends  # added this line
        ####

        ### distutils build_ext
        sources = ext.sources
        if sources is None or not isinstance(sources, (list, tuple)):
            msg = (
                f"in 'ext_modules' option (extension '{ext.name}'), "
                "'sources' must be present and must be "
                "a list of source filenames"
            )
            raise DistutilsSetupError(msg)
        # sort to make the resulting .so file build reproducible
        sources = sorted(sources)

        ext_path = self.get_ext_fullpath(ext.name)
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, self.get_ext_filename(ext.name), "newer")):  # updated this line
            log.info("skipping '%s' extension (up-to-date)", ext.name)
            self._skipped_extensions.add(ext)
            return
        log.info("building '%s' extension", ext.name)

        # First, scan the sources for SWIG definition files (.i), run
        # SWIG on 'em to create .c files, and modify the sources list
        # accordingly.
        sources = self.swig_sources(sources, ext)

        # Next, compile the source code to object files.

        # TODO: not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        objects = self.compiler.compile(
            sources,
            output_dir=self.build_temp,
            macros=macros,
            include_dirs=ext.include_dirs,
            debug=self.debug,
            extra_postargs=extra_args,
            depends=ext.depends,
        )

        # TODO: outdated variable, kept here in case third-part code
        # needs it.
        self._built_objects = objects[:]

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)

        self.compiler.link_shared_object(
            objects,
            ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language,
        )

    def copy_extensions_to_source(self) -> None:
        """Copy the compiled extensions to the source directory."""
        if not self.editable_mode or self.force:
            super().copy_extensions_to_source()
            return

        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            if ext in self._skipped_extensions:
                continue
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            if os.path.exists(regular_file) or not ext.optional:  # noqa: PTH110
                self.copy_file(regular_file, inplace_file, level=self.verbose)

            if ext._needs_stub:  # noqa: SLF001
                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                self._write_stub_file(inplace_stub, ext, compile=True)
                # Always compile stub and remove the original (leave the cache behind)
                # (this behaviour was observed in previous iterations of the code)


setup(
    ext_modules=[
        Extension(
            name="rs_example_lib.frf",
            sources=["rs_example_lib/frf.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c++",
        ),
        Extension(
            name="rs_example_lib.rss",
            sources=["rs_example_lib/rss.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c++",
        ),
        Extension(
            name="rs_example_lib.trapz",
            sources=["rs_example_lib/trapz.pyx"],
            include_dirs=[np.get_include(), "rs_example_lib/"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c++",
        ),
    ],
    cmdclass={"build_ext": BuildExtMod},
)
