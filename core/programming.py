from .essentials import rawprogram
from .gtypes import _modgpu_supported_gtypes

def program_with_basetypes(filename):
    kernels = {}
    for _sptgtype in _modgpu_supported_gtypes:
        typename = _sptgtype.name
        gputype = _sptgtype.gputype
        dtype = _sptgtype.dtype
        prg = rawprogram(filename, {"basetype": gputype})
        kernel = prg.matmul
        kernels[typename] = kernel
    return kernels