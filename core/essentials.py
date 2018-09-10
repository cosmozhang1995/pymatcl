import pyopencl as cl
import re

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

class Kernel:
    def __init__(self, clkernel):
        self.clkernel = clkernel

    def __call__(self, gsize, *args):
        clargs = [ctx, gsize, None] + args
        return apply(self.clkernel, clargs)

class Program:
    def __init__(self, clprogram):
        self.clprogram = clprogram
        kernel_names = clprogram.get_info(cl.program_info.KERNEL_NAMES).split(';')
        for kname in kernel_names:
            clkernel = getattr(clprogram, kname)
            setattr(self, kname, Kernel(clkernel))

def rawprogram(filename, replacements={}):
    fp = open(filename, 'r')
    filecontent = fp.read()
    fp.close()
    for pat in replacements:
        filecontent = re.sub(re.compile("\\b" + pat + "\\b"), replacements[pat], filecontent)
    prg = cl.Program(ctx, filecontent).build()
    return prg

def program(filename, replacements={}):
    return Program(rawprogram(filename, replacements=replacements))
