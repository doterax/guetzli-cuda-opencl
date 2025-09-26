@rem setupt windows var
call vcvars64.bat

@echo %1 --machine 64 or 32
@echo %2  -G 

set machine_num=%1%
set debug_opt=%2%

if "%machine_num%" == "" set machine_num=64

@echo Execute NVCC...

@echo nvcc -Xcompiler "/wd 4819" -I"./" -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=compute_75 -O3 --machine %machine_num% %debug_opt% -ptx -o clguetzli/clguetzli.cu.ptx%machine_num%  clguetzli\clguetzli.cu
nvcc -Xcompiler "/wd 4819" -I"./" -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=compute_75 -O3 --machine %machine_num% %debug_opt% -ptx -o clguetzli/clguetzli.cu.ptx%machine_num%  clguetzli\clguetzli.cu

python -c "import sys;a=sys.argv;open(a[2],'wb').write(('const unsigned char '+a[3]+'[] = {'+','.join([hex(b) for b in open(a[1],'rb').read()])+'};').encode('utf-8'))" clguetzli/clguetzli.cu.ptx%machine_num% clguetzli/clguetzli_cu_ptx.h clguetzli_cu%machine_num%
python -c "import sys;a=sys.argv;open(a[2],'wb').write(('const unsigned char '+a[3]+'[] = {'+','.join([hex(b) for b in open(a[1],'rb').read()])+'};').encode('utf-8'))" clguetzli/clguetzli.cl clguetzli/clguetzli_cl_src.h clguetzli_cl_src