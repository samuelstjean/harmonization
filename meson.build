project('_glmnet',
        ['c', 'fortran'],
        version : '0.2',
        meson_version: '>= 1.2.3',
        default_options : [
                            'warning_level=1',
                            'buildtype=release'
                          ])

numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_23_API_VERSION'

python_sources = [
    'harmonization/__init__.py',
    'harmonization/config.py',
    'harmonization/dictionary.py',
    'harmonization/elastic_net.py',
    'harmonization/recon.py',
    'harmonization/scripts.py',
    'harmonization/solver.py',
]

test_sources = [
  'harmonization/tests/test_config.py',
  'harmonization/tests/test_scripts.py',
  'harmonization/tests/datasets/config.yaml',
  'harmonization/tests/datasets/config_upscale.yaml',
  'harmonization/tests/datasets/subj1/dwi.nii.gz',
  'harmonization/tests/datasets/subj1/dwi_brain_mask.nii.gz',
  'harmonization/tests/datasets/subj1/dwi.bvals',
  'harmonization/tests/datasets/subj1/dwi.bvecs',
  'harmonization/tests/datasets/subj2/dwi.nii.gz',
  'harmonization/tests/datasets/subj2/dwi_brain_mask.nii.gz',
  'harmonization/tests/datasets/subj2/dwi.bvals',
  'harmonization/tests/datasets/subj2/dwi.bvecs',
  'harmonization/tests/datasets_bids/config_bids.yaml',
  'harmonization/tests/datasets_bids/config_upscale_bids.yaml',
  'harmonization/tests/datasets_bids/subj1-bids/dwi/subj1-dwi.nii.gz',
  'harmonization/tests/datasets_bids/subj1-bids/dwi/subj1-dwi_brain_mask.nii.gz',
  'harmonization/tests/datasets_bids/subj1-bids/dwi/subj1-dwi.bvals',
  'harmonization/tests/datasets_bids/subj1-bids/dwi/subj1-dwi.bvecs',
  'harmonization/tests/datasets_bids/subj2-bids/dwi/subj2-dwi.nii.gz',
  'harmonization/tests/datasets_bids/subj2-bids/dwi/subj2-dwi_brain_mask.nii.gz',
  'harmonization/tests/datasets_bids/subj2-bids/dwi/subj2-dwi.bvals',
  'harmonization/tests/datasets_bids/subj2-bids/dwi/subj2-dwi.bvecs',
]

fflags = ['-fdefault-real-8',
          '-ffixed-form',
          '-fPIC',
          '-flto',
          '-O2',
          '-funroll-loops',
          '-ffast-math',
          '-shared',
]

fc = meson.get_compiler('fortran')
add_project_arguments(fflags, language: 'fortran')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np,
                            compile_args: numpy_nodepr_api)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy, incdir_f2py)
# gh-25000
quadmath_dep = fc.find_library('quadmath', required: false)

py.extension_module('_glmnet',
                     [
                     '''glmnet/glmnet.f''',
                     '''glmnet/_glmnetmodule.c''',
                     '''glmnet/_glmnet-f2pywrappers.f''',
                     fortranobject_c
                     ],
                     include_directories: [
                     inc_np,
                     ],
                     dependencies : [
                     py_dep,
                     quadmath_dep,
                     ],
                     install : true,
                     subdir: 'harmonization')

py.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'harmonization'  # Folder relative to site-packages to install to
)

py.install_sources(
  test_sources,
  pure: false,             # Will be installed next to binaries
  # subdir: 'harmonization/tests',  # Folder relative to site-packages to install to
  preserve_path: true
)
