__kernel void opstep(int segments, int interval, int seglength, __global basetype *src, __global basetype *dest)
{
    int gsize_per_seg = (seglength + LOCAL_SIZE - 1) / LOCAL_SIZE;
    int gsize = gsize_per_seg * segments;
    int bid = get_group_id(0);
    int sid = bid / gsize_per_seg;
    int bid_in_seg = bid - sid * gsize_per_seg;
    int tid = get_local_id(0);
    int gid_in_seg = bid_in_seg * LOCAL_SIZE + tid;

    __global basetype *gdata = src;
    __local basetype *ldata[LOCAL_SIZE];

    if (gid_in_seg < seglength) ldata[tid] = gdata[sid * interval + gid_in_seg];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int counter = LOCAL_SIZE >> 1; counter != 0; counter = counter >> 1) {
        if (tid < counter) {
            basetype a = ldata[tid];
            basetype b = ldata[tid + counter];
            a = opexpr;
            ldata[tid] = a;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dest[bid] = ldata[0];
}