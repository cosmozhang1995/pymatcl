from .essentials import ctx, queue, cl

class TempBufferManager:
    def __init__(self):
        self.buffer = None
    def get_buffer(self, size):
        if self.buffer is None or self.buffer.size < size:
            self.buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=size)
        return self.buffer

tempbuffer_manager = TempBufferManager()

def tempbuffer(size):
    return tempbuffer_manager.get_buffer(size)
