import pyopencl as cl
from pyopencl.cffi_cl import RuntimeError as CLRuntimeError
import numpy as np
from .essentials import program, ctx
from .gtypes import _modgpu_supported_gtypes

class BaseOperatorForType:
    def __init__(self, clprg, name=None, datatype=None):
        self.kernel_base = clprg.op
        self.kernel_scalar = clprg.op_scalar
        self.kernel_byscalar = clprg.op_byscalar
        self.name = name
        self.datatype = datatype

class BaseOperator:
    def __init__(self, clprgs, name=None):
        self.name = name
        self.clprgs = clprgs
    def for_type(self, typename):
        return self.clprgs[typename]
    def support(self, typename):
        return typename in self.clprgs

def _build_base_operator(name, clexpr):
    clfuncs = dict()
    for gtype in _modgpu_supported_gtypes:
        typename = gtype.name
        gputype = gtype.gputype
        dtype = gtype.dtype
        try:
            prgcode = """
                __kernel void op(__global const """ + gputype + """ *a_g, __global const """ + gputype + """ *b_g, __global """ + gputype + """ *res_g)
                {
                    int gid = get_global_id(0);
                    res_g[gid] = """ + clexpr.replace("a", "a_g[gid]").replace("b", "b_g[gid]") + """;
                }
                __kernel void op_scalar(__global const """ + gputype + """ *a_g, const """ + gputype + """ b, __global """ + gputype + """ *res_g)
                {
                    int gid = get_global_id(0);
                    res_g[gid] = """ + clexpr.replace("a", "a_g[gid]").replace("b", "b") + """;
                }
                __kernel void op_byscalar(const """ + gputype + """ a, __global const """ + gputype + """ *b_g, __global """ + gputype + """ *res_g)
                {
                    int gid = get_group_id(0);
                    res_g[gid] = """ + clexpr.replace("a", "a").replace("b", "b_g[gid]") + """;
                }"""
            prg = cl.Program(ctx, prgcode).build()
            op_for_type = BaseOperatorForType(prg, name=name)
            op_for_type.kernel_scalar.set_scalar_arg_dtypes([None, dtype, None])
            op_for_type.kernel_byscalar.set_scalar_arg_dtypes([dtype, None, None])
            clfuncs[typename] = op_for_type
        except CLRuntimeError as e:
            pass
    return BaseOperator(clfuncs, name=name)

base_operator_add = _build_base_operator("add", "a + b")
base_operator_sub = _build_base_operator("sub", "a - b")
base_operator_mul = _build_base_operator("mul", "a * b")
base_operator_div = _build_base_operator("div", "a / b")
base_operator_eq = _build_base_operator("eq", "a == b")
base_operator_lt = _build_base_operator("lt", "a < b")
base_operator_le = _build_base_operator("le", "a <= b")
base_operator_gt = _build_base_operator("gt", "a > b")
base_operator_ge = _build_base_operator("ge", "a >= b")
base_operator_and = _build_base_operator("and", "a & b")
base_operator_or = _build_base_operator("or", "a | b")
base_operator_xor = _build_base_operator("xor", "a ^ b")
