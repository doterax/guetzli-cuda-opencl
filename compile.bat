@rem setupt windows var
call vcvars64.bat

@echo Execute NVCC...

@echo nvcc -Xcompiler "/wd 4819" -I"./" -use_fast_math -arch=compute_75 -O3 --machine 64 -G -ptx -o clguetzli/clguetzli.cu.ptx  clguetzli\clguetzli.cu
nvcc -Xcompiler "/wd 4819" -I"./" -use_fast_math -arch=compute_75 -O3 --machine 64 -G -ptx -o clguetzli/clguetzli.cu.ptx clguetzli\clguetzli.cu

minilzoc\bin\minilzoc clguetzli/clguetzli.cu.ptx clguetzli/clguetzli.cu.ptx.lzo
minilzoc\bin\minilzoc clguetzli/clguetzli.cl clguetzli/clguetzli.cl.lzo
minilzoc\bin\minilzoc clguetzli/clguetzli_amd.cl clguetzli/clguetzli_amd.cl.lzo


python format_header.py clguetzli/clguetzli.cu.ptx.lzo clguetzli/clguetzli_cu_ptx.h clguetzli_cu64_lzo
python format_header.py clguetzli/clguetzli.cl.lzo clguetzli/clguetzli_cl_src.h clguetzli_cl_src_lzo
python format_header.py clguetzli/clguetzli_amd.cl.lzo clguetzli/clguetzli_cl_amd_src.h clguetzli_cl_amd_src_lzo
