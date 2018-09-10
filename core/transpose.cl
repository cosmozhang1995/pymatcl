__kernel void matmul(__global const basetype *in_g, __global basetype *out_g)
{
    int outrows = get_global_size(0);
    int outcols = get_global_size(1);
    int inrows = outcols;
    int incols = outrows;
    int ioutr = get_global_id(0);
    int ioutc = get_global_id(1);
    int iinr = ioutc;
    int iinc = ioutr;
    int iout = ioutr * outcols + ioutc;
    int iin = iinr * incols + iinc;
    out_g[iout] = in_g[iin];
}