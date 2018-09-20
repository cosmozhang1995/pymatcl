import numpy as np
from .essentials import ctx, queue, cl
from .tempbuffer import tempbuffer

REDUCE_LOCAL_SIZE = ctx.get_platforms()[0].get_devices()[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

_script_dir = os.path.split(os.path.realpath(__file__))[0]
_srccode_file = open(os.path.join(_script_dir, "reduce.cl"))
_srccode = _srccode_file.read()
_srccode_file.close()

def _compile_kernel_for_op(opexpr):
    replacements = {"LOCAL_SIZE": REDUCE_LOCAL_SIZE, "opexpr": opexpr}
    retkernel = program_with_basetypes(source=_srccode, replacements=replacements).opstep
    for k in retkernel:
        retkernel[k].set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None])
_replacements["opexpr"] = "a + b"
_kernels_sum = program_with_basetypes(source=_srccode, replacements=_replacements).opstep

_replacements["opexpr"] = "a * b"
_kernels_prod = program_with_basetypes(source=_srccode, replacements=_replacements).opstep

def _reduce_op(kernel, inmat, dimension):
    shape = inmat.shape
    segments = prod(shape[0:dimension])
    interval = prod(shape[(dimension+1):])
    seglength = shape[dimension]
    kernel = kernel[inmat.gtype.name]
    tmpbuf = tempbuffer(inmat.size)
