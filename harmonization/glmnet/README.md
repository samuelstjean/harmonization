These files are the fortran source, taken from glmnet here 
+ https://cran.r-project.org/web/packages/glmnet/index.html

To generate the pyf wrapper, use 
+ f2py -c --fcompiler=gnu95 --f77flags='-fdefault-real-8' --f90flags='-fdefault-real-8' glmnet.pyf glmnet.f

