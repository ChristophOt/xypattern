from pathlib import Path
import shutil
from Cython.Build import cythonize, build_ext
from setuptools.dist import Distribution

extensions = ["xypattern/util/smooth_bruckner.pyx"]
ext_modules = cythonize(extensions, compiler_directives={'language_level': 3})
dist = Distribution({'ext_modules': ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    output = Path(output)
    print("Build output: ", output)
    relative_path = output.relative_to(cmd.build_lib)
    shutil.copyfile(output, relative_path)
