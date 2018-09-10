import numpy as np


class _modgpu_supported_dtype_mapping_item:
    def __init__(self, _elemsize, _dtypecls, _gputype, _pytype):
        self.name = None
        self.elemsize = _elemsize
        self.dtype = np.dtype(_dtypecls)
        self.dtypecls = _dtypecls
        self.gputype = _gputype
        self.pytype = _pytype
_modgpu_supported_gtype_mappings = {
    "float32": _modgpu_supported_dtype_mapping_item(4, np.float32, "float",         float),
    "int32":   _modgpu_supported_dtype_mapping_item(4, np.int32,   "int",           int),
    "int8":    _modgpu_supported_dtype_mapping_item(1, np.int8,    "char",          int),
    "uint32":  _modgpu_supported_dtype_mapping_item(4, np.uint32,  "unsigned int",  int),
    "uint8":   _modgpu_supported_dtype_mapping_item(1, np.uint8,   "unsigned char", int)
}
for name in _modgpu_supported_gtype_mappings:
    _modgpu_supported_gtype_mappings[name].name = name
_modgpu_supported_gtype_dtypeclses = list(map(lambda x: _modgpu_supported_gtype_mappings[x].dtypecls, _modgpu_supported_gtype_mappings))

def _modgpu_map_type(given):
    item = None
    if isinstance(given, np.dtype):
        attrname = "dtype"
    elif given in _modgpu_supported_gtype_dtypeclses:
        given = np.dtype(given)
        attrname = "dtype"
    elif isinstance(given, str):
        attrname = "gputype"
        if given in _modgpu_supported_gtype_mappings:
            item = _modgpu_supported_gtype_mappings[given]
    elif isinstance(given, float):
        attrname = "pytype"
    elif isinstance(given, int):
        attrname = "pytype"
    else:
        raise ValueError("Not recognized given value type")
    if item is None:
        for name in _modgpu_supported_gtype_mappings:
            item = _modgpu_supported_gtype_mappings[name]
            if getattr(item, attrname) == given:
                return item
    else:
        return item
    return None

class gtype:
    @property
    def name(self):
        return self._name
    @property
    def elemsize(self):
        return self._elemsize
    @property
    def dtype(self):
        return self._dtype
    @property
    def gputype(self):
        return self._gputype
    @property
    def type(self):
        return self._pytype
    @property
    def pytype(self):
        return self._pytype
        
    def __init__(self, init):
        if isinstance(init, gtype):
            self._name = init.name
            self._elemsize = init.elemsize
            self._dtype = init.dtype
            self._gputype = init.gputype
            self._pytype = init.pytype
        else:
            init = _modgpu_map_type(init)
            if init is None:
                raise ValueError("Not supported type given.")
            self._name = init.name
            self._elemsize = init.elemsize
            self._dtype = init.dtype
            self._gputype = init.gputype
            self._pytype = init.pytype

    def __eq__(self, other):
        if isinstance(other, gtype):
            return self.name == other.name
        else:
            try:
                other = gtype(other)
                return self == other
            except Exception as e:
                return False

    def __str__(self):
        return self.name

float32 = gtype(np.float32)
int32 = gtype(np.int32)
int8 = gtype(np.int8)
uint32 = gtype(np.uint32)
uint8 = gtype(np.uint8)

_modgpu_supported_gtypes = list(map(lambda x: gtype(x), _modgpu_supported_gtype_mappings))

def is_gtype_valid(the_dtype):
    try:
        gtype(the_dtype)
        return True
    except Exception as e:
        return False

