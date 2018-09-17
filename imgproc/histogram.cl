// gin.size = size
// gtemp.size = global_size * local_size
// global_size and 
__kernel void histogram(int size, __global basetype *gin, __global int *gtemp)
{
    int gid = get_global_id(0);
    int gsize = get_global_size(0);
    int lsize0 = get_local_size(0);
    int lsize1 = get_local_size(1);
    int lsize = lsize0 * lsize1;
    int lid = get_local_id(0) * lsize1 + get_local_id(1);

    // Compute histogram for each `steps` piece
    int piecestart = gid * steps;
    int pieceend = piecestart + steps;
    if (pieceend > size) pieceend = size;
    for (int i = piecestart; i < pieceend; i++) {
        int val = (int)(pin[i]);
        if (val >= 0 && val <= steps) {
            temp[val] += 1;
        }
    }

    // Reduce to calculate the final temp
    int idx1 = gid * lsize + lid;
    int p = 0;
    int gap = 1;
    while () {
        int idx2 = idx + gap;
        
    }
}