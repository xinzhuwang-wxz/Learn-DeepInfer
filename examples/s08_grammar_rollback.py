"""Tiny character grammar accepts exactly 'true' or 'false'. NOT xgrammar/JSON schema."""
class Matcher:
    words = ('true', 'false')
    def __init__(self):
        self.prefix = ''

    def allowed(self):
        result = {w[len(self.prefix)] for w in self.words if w.startswith(self.prefix) and len(w)>len(self.prefix)}
        if self.prefix in self.words:
            result.add('<eos>')
        return result

    def accept(self, token):
        if token not in self.allowed():
            return False
        if token != '<eos>':
            self.prefix += token
        return True

    def rollback(self, steps):
        if not 0 <= steps <= len(self.prefix):
            raise ValueError('rollback exceeds history')
        if steps:
            self.prefix = self.prefix[:-steps]

if __name__ == '__main__':
    matcher = Matcher()
    for token in 'tru':
        matcher.accept(token)
    print('after tru:', matcher.allowed())
    matcher.rollback(2)
    print('after rollback 2:', matcher.prefix, matcher.allowed())
