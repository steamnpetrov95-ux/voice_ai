"""Seekable, bounded HTTPS range reader for inspecting large model archives."""

from collections import OrderedDict
import io
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size = url, size
        self.position = 0
        self.block_size = 8 * 1024 * 1024
        self.cache = OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(self.size, start + self.block_size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise OSError("Server did not honor the bounded byte range")
            block = response.read(end - start + 2)
        if len(block) != end - start + 1:
            raise OSError("Incomplete archive range")
        self.cache[index] = block
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return block

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size > 64 * 1024 * 1024:
            raise OSError("Use streaming reads for large archive entries")
        result = bytearray()
        while size > 0:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            count = min(size, len(block) - offset)
            if count <= 0:
                raise OSError("Unexpected end of archive block")
            result.extend(block[offset:offset + count])
            self.position += count
            size -= count
        return bytes(result)
