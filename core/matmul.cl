__kernel void matmul(const int len, __global const basetype *a_g, __global const basetype *b_g, __global basetype *res_g)
{
    int rows = get_global_size(0);
    int cols = get_global_size(1);
    int ir = get_global_id(0);
    int ic = get_global_id(1);
    int rowsa = rows;
    int colsa = len;
    int rowsb = len;
    int colsb = cols;
    int gid0a = ir * colsa;
    int stepa = 1;
    int gid0b = ic;
    int stepb = colsb;
    int i = 0;
    int ia = gid0a;
    int ib = gid0b;
    int ires = ir * cols + ic;
    basetype sum = 0;
    for (i = 0; i < len; i++) {
        sum = sum + a_g[ia] * b_g[ib];
        ia += stepa;
        ib += stepb;
    }
    res_g[ires] = sum;
}