#!/usr/bin/env python3
"""Turn a macOS `sample` call-graph dump into the profile table.

`sample` renders a call tree with a line drawing whose characters its man page
calls arbitrary, so the parser uses the *column of the count* as the level, and
derives self time as inclusive - sum(children).  It also rejects an attachment
that would over-count its parent, because `sample` appends a flat "leaf
subtree" summary after the last thread that otherwise nests under it.  A
correct parse reports 0 over-counted nodes; anything else means the numbers are
wrong, so it is printed prominently.

Idle pool waits (`__psynch_cvwait` and friends) are reported as a separate
line and excluded from the work table: the joint-plan walk is serial, so
7 of 8 workers sit in `condition_variable::wait` for most of the encode.

Usage:
  sample_to_table.py SAMPLE.txt [--top N] [--markdown]
"""
import argparse
import re
from collections import defaultdict

LINE = re.compile(r'^(?P<indent>[+!:|\s]+)(?P<count>\d+)\s+(?P<rest>.*)$')
THREAD = re.compile(r'^\s*(?P<count>\d+)\s+Thread_')
FRAME = re.compile(r'^(?P<sym>.*?)\s+\(in ')

IDLE = {
    '__psynch_cvwait', '__psynch_cvsignal', '__psynch_mutexwait',
    'std::condition_variable::wait', '_pthread_cond_wait',
}

# Subsystem rollup.  A function belongs to the first group whose prefix it
# matches, so order matters for overlapping names.
GROUPS = [
    ('RDOQ model + trellis', ('rdoq::Model::', 'rdoq::rdoq_quantize',
                              'rdoq::av2::build_scan', 'rdoq::av2::Model::')),
    ('transform pair', ('txv::forward_block', 'txv::inverse_block')),
    ('tx search driver', ('txsearch::', 'txpartition::')),
    ('libm log2', ('log2',)),
    ('libc alloc / mem', ('_platform_memset', '_platform_memmove', '__bzero',
                          'operator new', 'operator delete', 'malloc',
                          'free', '_xzm_', 'xzm_', '_tlv_get_addr')),
    ('intra prediction', ('intra::', 'intra_search::', 'cfl::')),
    ('perceptual metric', ('eff_perc::', 'pipe_perc::', 'pipe_ba::')),
    ('exact rate interval', ('rate_interval::', 'plan_rate_bits',
                             'residual::')),
    ('motion / MC', ('inter::mc::', 'inter::adv::', 'me::', 'primitives::',
                     'multiref::', 'merge::')),
    ('syntax writer / entropy', ('syntax::',)),
]


def short(sym):
    s = sym.strip()
    s = s.replace('std::__function::__func<', '')
    out, depth = [], 0
    for ch in s:
        if ch == '<':
            depth += 1
            continue
        if ch == '>':
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    s = ''.join(out).strip()
    paren = s.find('(')
    if paren > 0:
        s = s[:paren]
    for a, b in (('av2enc::video::txtsearch::', 'txsearch::'),
                 ('av2enc::video::vf_detail::', 'vf::'),
                 ('av2enc::rdoq::av2::Model::', 'rdoq::Model::'),
                 ('av2enc::rdoq::', 'rdoq::'),
                 ('av2enc::video::', ''),
                 ('av2enc::primitives::(anonymous namespace)::', 'primitives::'),
                 ('av2enc::', ''),
                 ('std::__1::', 'std::'),
                 ('(anonymous namespace)::', '')):
        s = s.replace(a, b)
    s = re.sub(r'::\$_?\d+::operator\(\)', '::lambda', s)
    s = re.sub(r'::operator\(\)', '', s)
    return s.strip() or '?'


class Node:
    __slots__ = ('name', 'count', 'children', '_depth')

    def __init__(self, name, count):
        self.name, self.count, self.children, self._depth = name, count, [], 0


def parse(path):
    roots, stack = [], []
    for raw in open(path, errors='replace'):
        tm = THREAD.match(raw)
        if tm:
            cur = Node('~thread', int(tm.group('count')))
            roots.append(cur)
            stack = [cur]
            continue
        m = LINE.match(raw.rstrip('\n'))
        if not m or not stack:
            continue
        fm = FRAME.match(m.group('rest'))
        if not fm:
            continue
        depth = (len(m.group('indent')) - 6) // 2   # 2-char cell per level
        count = int(m.group('count'))
        node = Node(short(fm.group('sym')), count)
        while len(stack) > 1 and (stack[-1]._depth >= depth or
                                  stack[-1].count < count):
            stack.pop()
        parent = stack[-1]
        if sum(c.count for c in parent.children) + count > parent.count:
            continue          # sample's trailing flat summary
        node._depth = depth
        parent.children.append(node)
        stack.append(node)
    return roots


def collect(node, prefix, out, problems):
    child_sum = sum(c.count for c in node.children)
    if child_sum > node.count:
        problems.append((node.name, node.count, child_sum))
    self_time = node.count - child_sum
    if self_time > 0:
        out[prefix + (node.name,)] += self_time
    for c in node.children:
        collect(c, prefix + (node.name,), out, problems)


def group_of(fn):
    for name, prefixes in GROUPS:
        if any(fn == p or fn.startswith(p) for p in prefixes):
            return name
    return 'other / unattributed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sample')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--markdown', action='store_true')
    a = ap.parse_args()

    roots = parse(a.sample)
    stacks, problems = defaultdict(int), []
    for r in roots:
        collect(r, (), stacks, problems)
    all_total = sum(stacks.values())
    work = {k: v for k, v in stacks.items()
            if not (k and k[-1] in IDLE)}
    total = sum(work.values())

    bar = '|' if a.markdown else ' '
    def row(*cells):
        if a.markdown:
            print('| ' + ' | '.join(cells) + ' |')
        else:
            print('  '.join(f'{c:>28}' if i else f'{c:<28}'
                            for i, c in enumerate(cells)))

    print(f'over-counted nodes: {len(problems)}'
          f'{" (PARSE IS WRONG)" if problems else ""}')
    print(f'all-thread samples: {all_total:,}')
    print(f'idle samples:       {all_total - total:,}'
          f' ({100.0 * (all_total - total) / max(all_total, 1):.1f}%)')
    print(f'work samples:       {total:,}')

    self_by = defaultdict(int)
    for path, n in work.items():
        self_by[path[-1]] += n

    print(f'\ntop {a.top} by self time')
    if a.markdown:
        row('#', 'function', 'samples', 'share')
        row('---:', '---', '---:', '---:')
    for i, (fn, n) in enumerate(
            sorted(self_by.items(), key=lambda kv: -kv[1])[:a.top], 1):
        if a.markdown:
            row(str(i), f'`{fn}`', f'{n:,}', f'{100.0 * n / total:.1f} %')
        else:
            print(f'{i:3}. {100.0 * n / total:5.1f}%  {n:6,}  {fn}')

    print('\nby subsystem (self time)')
    if a.markdown:
        row('group', 'samples', 'share')
        row('---', '---:', '---:')
    g = defaultdict(int)
    for fn, n in self_by.items():
        g[group_of(fn)] += n
    for name, n in sorted(g.items(), key=lambda kv: -kv[1]):
        if a.markdown:
            row(name, f'{n:,}', f'{100.0 * n / total:.1f} %')
        else:
            print(f'  {100.0 * n / total:5.1f}%  {n:6,}  {name}')


if __name__ == '__main__':
    main()
