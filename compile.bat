@rem setupt windows var
call vcvars64.bat

@echo Execute NVCC...

@echo nvcc -Xcompiler "/wd 4819" -I"./" -use_fast_math -arch=compute_75 -O3 --machine 64 -G -ptx -o clguetzli/clguetzli.cu.ptx  clguetzli\clguetzli.cu
nvcc -Xcompiler "/wd 4819" -I"./" -use_fast_math -arch=compute_75 -O3 --machine 64 -G -ptx -o clguetzli/clguetzli.cu.ptx clguetzli\clguetzli.cu


python format_header.py clguetzli/clguetzli.cu.ptx clguetzli/clguetzli_cu_ptx.h clguetzli_cu64
python format_header.py clguetzli/clguetzli.cl clguetzli/clguetzli_cl_src.h clguetzli_cl_src
python format_header.py clguetzli/clguetzli_amd.cl clguetzli/clguetzli_cl_amd_src.h clguetzli_cl_amd_src
