"""Reference-counted block teaching model. Zero is reserved; no KV tensor allocation."""
class Blocks:
    def __init__(self, total):
        if total < 2:
            raise ValueError('need sentinel plus at least one usable block')
        self.refs = [0] * total

    def status(self, count):
        if count < 0:
            raise ValueError('negative allocation')
        if count >= len(self.refs):
            return 'NEVER'
        free = sum(n == 0 for n in self.refs[1:])
        return 'OK' if free >= count else 'LATER'

    def allocate(self, count):
        if self.status(count) != 'OK':
            raise ValueError(self.status(count))
        ids = [i for i in range(1, len(self.refs)) if self.refs[i] == 0][:count]
        for i in ids:
            self.refs[i] = 1
        return ids

    def pin(self, i):
        self._live(i)
        self.refs[i] += 1

    def free(self, i):
        self._live(i)
        self.refs[i] -= 1

    def _live(self, i):
        if not 0 < i < len(self.refs) or self.refs[i] <= 0:
            raise ValueError('not a live block')

if __name__ == '__main__':
    pool = Blocks(4)
    block = pool.allocate(1)[0]
    pool.pin(block)
    pool.free(block)
    print('after one owner releases:', pool.refs)
    pool.free(block)
    print('after last owner releases:', pool.refs)
