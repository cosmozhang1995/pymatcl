from .essentials import rawprogram
from .gtypes import _modgpu_supported_gtypes

def program_with_basetypes(filename=None, source=None, replacements={}):
    kernels = {}
    theops = {}
    for k in replacements:
        theops[k] = replacements[k]
    for _sptgtype in _modgpu_supported_gtypes:
        typename = _sptgtype.name
        gputype = _sptgtype.gputype
        dtype = _sptgtype.dtype
        theops["basetype"] = gputype
        prg = rawprogram(filename=filename, source=source, replacements=theops)
        kernel = prg.matmul
        kernels[typename] = kernel
    return kernels