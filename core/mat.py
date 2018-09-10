import pyopencl as cl
import numpy as np
import os
from .essentials import rawprogram, ctx, queue
from .gtypes import gtype, is_gtype_valid
from .gtypes import _modgpu_supported_gtypes
from .base_operator import \
    base_operator_add,\
    base_operator_sub,\
    base_operator_mul,\
    base_operator_div,\
    base_operator_eq,\
    base_operator_lt,\
    base_operator_le,\
    base_operator_gt,\
    base_operator_ge,\
    base_operator_and,\
    base_operator_or,\
    base_operator_xor
from .mathutils import prod

_script_dir = os.path.split(os.path.realpath(__file__))[0]

_dtype_mapping = ((np.float64, np.float32), (np.int64, np.int32))
_dtype_mapping = tuple(map(lambda x: tuple(map(lambda y: np.dtype(y), x)), _dtype_mapping))
_dtype_disabled = tuple(map(lambda x: x[0], _dtype_mapping))

def program_with_basetypes(filename):
    kernels = {}
    for _sptgtype in _modgpu_supported_gtypes:
        typename = _sptgtype.name
        gputype = _sptgtype.gputype
        dtype = _sptgtype.dtype
        prg = rawprogram(os.path.join(_script_dir, filename), {"basetype": gputype})
        kernel = prg.matmul
        kernels[typename] = kernel
    return kernels

_kernel_matmul = program_with_basetypes("matmul.cl")
for k in _kernel_matmul:
    _kernel_matmul[k].set_scalar_arg_dtypes([np.int32, None, None, None])
_kernel_transpose = program_with_basetypes("transpose.cl")


class Mat:
    @property
    def shape(self):
        return self._shape
    @property
    def dtype(self):
        return self._gtype
    @property
    def gtype(self):
        return self._gtype
    @property
    def buffer(self):
        return self._buffer
    @property
    def ndims(self):
        return len(self._shape)
    
    
    
    def __init__(self, init=None, shape=None, datatype=None, _sharing_parent=None):
        if _sharing_parent is None:
            # No parent sharing memory, create new buffer
            self._sharing_parent = None
            if not datatype is None and not is_gtype_valid(datatype):
                raise ValueError("Type %s is not allowed" % str(gtype(datatype)))
            if isinstance(init, Mat):
                # clone constructor
                self._shape = init.shape
                self._gtype = init.gtype
                self._buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size = prod(self.shape) * self.gtype.elemsize)
                cl.enqueue_copy(queue, self._buffer, init.buffer)
            else:
                # construct with host buffer or/and specified shape and datatype
                if init is None:
                    hostbuf = None
                else:
                    hostbuf = np.array(init)
                    for _dtype_mapping_item in _dtype_mapping:
                        if hostbuf.dtype == _dtype_mapping_item[0]:
                            hostbuf = hostbuf.astype(_dtype_mapping_item[1])
                if hostbuf is None:
                    self._shape = shape or tuple()
                    self._gtype = gtype(datatype)
                    hostbuf = np.empty(self._shape, dtype=self._gtype.dtype)
                if len(hostbuf.shape) == 0:
                    self._shape = shape or tuple()
                    self._gtype = gtype(datatype or hostbuf.dtype)
                    hostbuf = np.zeros(self._shape, dtype=self._gtype.dtype) + hostbuf.astype(self._gtype)
                else:
                    self._shape = hostbuf.shape
                    self._gtype = gtype(datatype or hostbuf.dtype)
                    hostbuf = hostbuf.astype(self._gtype)
                self._buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)
        else:
            # construct with a parent, share the buffer of the parent
            self._sharing_parent = _sharing_parent
            self._shape = _sharing_parent.shape
            self._gtype = _sharing_parent.gtype
            self._buffer = _sharing_parent.buffer
        self._sharing_children = []

    # Before changes the data of this Mat instance, we must detach it from the memory-sharing relations
    def _detach_sharing(self):
        if not self._sharing_parent is None:
            self._sharing_parent._sharing_children = list(filter(lambda x: x != self, self._sharing_parent._sharing_children)) + self._sharing_children
            for child in self._sharing_children:
                child._sharing_parent = self._sharing_parent
                child._buffer = self._sharing_parent._buffer
                for grandchild in child._sharing_children:
                    grandchild._buffer = child._buffer
            self._sharing_parent = None
            self._sharing_children = []
            _old_buffer = self._buffer
            self._buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size = prod(self.shape) * self.gtype.elemsize)
            cl.enqueue_copy(queue, self._buffer, _old_buffer)
        elif len(self._sharing_children) > 0:
            newparent = self._sharing_children[0]
            newparent._sharing_parent = None
            newparent._buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size = prod(self.shape) * self.gtype.elemsize)
            cl.enqueue_copy(queue, newparent._buffer, self._buffer)
            for grandchild in newparent._sharing_children:
                grandchild._buffer = newparent._buffer
            for child in self._sharing_children[1:]:
                child._sharing_parent = newparent
                child._buffer = newparent._buffer
                for grandchild in child._sharing_children:
                    grandchild._buffer = child._buffer
            self._sharing_children = []

    def gather(self):
        hostbuf = np.empty(self._shape, dtype=self._gtype.dtype)
        cl.enqueue_copy(queue, hostbuf, self._buffer)
        return hostbuf

    def _check_other_and_take_op(base_op, va, vb, vo=None):
        ismat_a = isinstance(va, Mat)
        ismat_b = isinstance(vb, Mat)
        pytype_a = gtype(va.gtype).pytype if ismat_a else type(va)
        pytype_b = gtype(vb.gtype).pytype if ismat_b else type(vb)
        pytype_a_str = str(pytype_a).replace("<class '", "").replace("'>", "")
        pytype_b_str = str(pytype_b).replace("<class '", "").replace("'>", "")
        if ismat_a and ismat_b:
            if va.shape != vb.shape:
                raise ValueError("Matrix dimensions or shapes not match")
            elif va.gtype != vb.gtype:
                raise ValueError("Matrix types not match")
            else:
                kernel_type = va.gtype.name
                if not base_op.support(kernel_type):
                    raise ValueError("Operation '%s' is not permitted on data with type: %s" % (base_op.name, kernel_type))
                kernel = base_op.for_type(kernel_type).kernel_base
                if vo is None:
                    vo = Mat(shape=va.shape, datatype=va.gtype)
                kernel(queue, (prod(va.shape),), None, va._buffer, vb._buffer, vo._buffer)
                return vo
        elif ismat_a and not ismat_b:
            if not pytype_a == pytype_b:
                raise ValueError("Left matrix with type '%s' is not competible with right scalar with type '%s'" % (str(va.gtype), pytype_b_str))
            else:
                kernel_type = va.gtype.name
                if not base_op.support(kernel_type):
                    raise ValueError("Operation '%s' is not permitted on data with type: %s" % (base_op.name, kernel_type))
                kernel = base_op.for_type(kernel_type).kernel_scalar
                if vo is None:
                    vo = Mat(shape=va.shape, datatype=va.gtype)
                kernel(queue, (prod(va.shape),), None, va._buffer, vb, vo._buffer)
                return vo
        elif not ismat_a and ismat_b:
            if not pytype_a == pytype_b:
                raise ValueError("Left scalar with type '%s' is not competible with right matrix with type '%s'" % (pytype_a_str, str(vb.gtype)))
            else:
                kernel_type = vb.gtype.name
                if not base_op.support(kernel_type):
                    raise ValueError("Operation '%s' is not permitted on data with type: %s" % (base_op.name, kernel_type))
                kernel = base_op.for_type(kernel_type).kernel_byscalar
                if vo is None:
                    vo = Mat(shape=vb.shape, datatype=vb.gtype)
                kernel(queue, (prod(vb.shape),), None, va, vb._buffer, vo._buffer)
                return vo
        else:
            raise ValueError("Neither is a matrix")
        return None

    def __add__(self, other):
        return Mat._check_other_and_take_op(base_operator_add, self, other)
    def __radd__(self, other):
        return Mat._check_other_and_take_op(base_operator_add, other, self)
    def __iadd__(self, other):
        self._detach_sharing()
        return Mat._check_other_and_take_op(base_operator_add, self, other, self)

    def __sub__(self, other):
        return Mat._check_other_and_take_op(base_operator_sub, self, other)
    def __rsub__(self, other):
        return Mat._check_other_and_take_op(base_operator_sub, other, self)
    def __isub__(self, other):
        self._detach_sharing()
        return Mat._check_other_and_take_op(base_operator_sub, self, other, self)

    def __mul__(self, other):
        return Mat._check_other_and_take_op(base_operator_mul, self, other)
    def __rmul__(self, other):
        return Mat._check_other_and_take_op(base_operator_mul, other, self)
    def __imul__(self, other):
        self._detach_sharing()
        return Mat._check_other_and_take_op(base_operator_mul, self, other, self)

    def __div__(self, other):
        return Mat._check_other_and_take_op(base_operator_div, self, other)
    def __rdiv__(self, other):
        return Mat._check_other_and_take_op(base_operator_div, other, self)
    def __idiv__(self, other):
        self._detach_sharing()
        return Mat._check_other_and_take_op(base_operator_div, self, other, self)


    def __and__(self, other):
        return Mat._check_other_and_take_op(base_operator_and, self, other)
    def __rand__(self, other):
        return Mat._check_other_and_take_op(base_operator_and, other, self)
    def __iand__(self, other):
        self._detach_sharing()
        return Mat._check_other_and_take_op(base_operator_and, self, other, self)

    def __or__(self, other):
        return Mat._check_other_and_take_op(base_operator_or, self, other)
    def __ror__(self, other):
        return Mat._check_other_and_take_op(base_operator_or, other, self)
    def __ior__(self, other):
        self._detach_sharing()
        return Mat._check_other_and_take_op(base_operator_or, self, other, self)


    def __xor__(self, other):
        return Mat._check_other_and_take_op(base_operator_xor, self, other)
    def __rxor__(self, other):
        return Mat._check_other_and_take_op(base_operator_xor, other, self)
    def __ixor__(self, other):
        self._detach_sharing()
        return Mat._check_other_and_take_op(base_operator_xor, self, other, self)

    def __eq__(self, other):
        return Mat._check_other_and_take_op(base_operator_eq, self, other)
    def __gt__(self, other):
        return Mat._check_other_and_take_op(base_operator_gt, self, other)
    def __ge__(self, other):
        return Mat._check_other_and_take_op(base_operator_ge, self, other)
    def __lt__(self, other):
        return Mat._check_other_and_take_op(base_operator_lt, self, other)
    def __le__(self, other):
        return Mat._check_other_and_take_op(base_operator_le, self, other)

    def _matmul(va, vb, vo=None):
        if len(va.shape) != 2 or len(vb.shape) != 2:
            raise ValueError("Matrix multiplication is not supported on non-2D matrices")
        if va.shape[1] != vb.shape[0]:
            raise ValueError("Matrix multiplication error: Left matrix size %s is not competible with right matrix size %s" % (va.shape, vb.shape))
        if va.gtype != vb.gtype:
            raise ValueError("Matrix multiplication error: Left matrix type %s is not competible with right matrix type %s" % (va.gtype, vb.gtype))
        length = va.shape[1]
        voshape = (va.shape[0], vb.shape[1])
        votype = va.gtype
        if vo is None:
            vo = Mat(shape=voshape, datatype=votype)
        else:
            if voshape != vo.shape:
                raise ValueError("Matrix multiplication error: Output matrix size %s not valid. Valid size is %s" % (vo.shape, voshape))
            if votype != vo.gtype:
                raise ValueError("Matrix multiplication error: Output matrix type %s not valid. Valid type is %s" % (vo.gtype, votype))
        kernel = _kernel_matmul[votype.name]
        kernel(queue, voshape, None, length, va.buffer, vb.buffer, vo.buffer)
        return vo

    def __matmul__(self, other):
        return Mat._matmul(self, other)
    def __rmatmul__(self, other):
        return Mat._matmul(other, self)
    def __imatmul__(self, other):
        raise NotImplementedError("Matrix multiplication have no augmented version")

    def reshape(self, newshape):
        vo = Mat(_sharing_parent=self)
        newshape = tuple(newshape)
        oldsize = prod(vo._shape)
        newsize = prod(newshape)
        if oldsize != newsize:
            raise ValueError("Reshape must not change the data size of the matrix")
        vo._shape = newshape
        return vo

    def transpose(self):
        if self.ndims != 2:
            raise NotImplementedError("Transpose of non-2D matrix is not supported")
        vo = Mat(datatype=self._gtype, shape=(self._shape[1], self._shape[0]))
        kernel = _kernel_transpose[vo.gtype.name]
        kernel(queue, vo.shape, None, self.buffer, vo.buffer)
        return vo

    @property
    def T(self):
        return self.transpose()
    


