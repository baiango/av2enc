#!/usr/bin/env python3
"""Build a flame-graph SVG from a macOS `sample` call-graph dump.

`sample` renders the call tree with a line drawing, not with indentation
levels, so depth cannot be read off the prefix directly: sibling lines repeat
the same prefix (e.g. two lines both at `+   ! : |`).  What *is* reliable is the
'inclusive count' on every line plus the tree's drawing order:

  * a line nests under the previous one when its parser-depth is greater,
  * otherwise it is a sibling of some open ancestor.

`sample` also annotates the thread's root with its own total, so the parse
produces exactly one tree per thread, and the tree is validated by requiring
every node's inclusive count to be >= the sum of its children's.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r'^(?P<indent>[+!:|\s]+)(?P<count>\d+)\s+(?P<rest>.*)$')
THREAD = re.compile(r'^\s*(?P<count>\d+)\s+Thread_')
FRAME = re.compile(r'^(?P<sym>.*?)\s+\(in ')


def short(sym):
    s = sym.strip()
    s = s.replace('std::__function::__func<', 'fn::')
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
    for a, b in (
        ('av2enc::video::InterFrameEncoder::', 'IFE::'),
        ('av2enc::video::txtsearch::', 'txsearch::'),
        ('av2enc::video::vf_detail::', 'vf::'),
        ('av2enc::video::', ''),
        ('av2enc::rdoq::av2::Model::', 'rdoq::Model::'),
        ('av2enc::rdoq::', 'rdoq::'),
        ('av2enc::txv::', 'txv::'),
        ('av2enc::intra::', 'intra::'),
        ('av2enc::inter::', 'inter::'),
        ('av2enc::parallel::', 'par::'),
        ('av2enc::me::', 'me::'),
        ('av2enc::primitives::(anonymous namespace)::', 'prim::'),
        ('av2enc::video::(anonymous namespace)::', ''),
        ('av2enc::', ''),
        ('std::__1::', 'std::'),
        ('(anonymous namespace)::', ''),
    ):
        s = s.replace(a, b)
    s = re.sub(r'::\$_?\d+::operator\(\)', '::lambda', s)
    s = re.sub(r'::operator\(\)', '', s)
    s = re.sub(r"::'lambda'", '::lambda', s)
    s = re.sub(r'^std::', '', s)
    return s.strip() or '?'


class Node:
    __slots__ = ('name', 'count', 'children', '_depth')

    def __init__(self, name, count):
        self.name = name
        self.count = count
        self.children = []
        self._depth = 0


def marker_depth(indent):
    # `sample` draws the tree with a 2-character cell per level (a drawing
    # char plus a space), and the man page says the chars themselves are
    # arbitrary.  The *column* of the count is what encodes the level.
    return (len(indent) - 6) // 2


def parse(path):
    roots, stack, current = [], [], None
    for raw in open(path, errors='replace'):
        tm = THREAD.match(raw)
        if tm:
            current = Node(f'~thread', int(tm.group('count')))
            roots.append(current)
            stack = [current]
            continue
        m = LINE.match(raw.rstrip('\n'))
        if not m or current is None:
            continue
        fm = FRAME.match(m.group('rest'))
        if not fm:
            continue
        indent, count = m.group('indent'), int(m.group('count'))
        depth = marker_depth(indent)
        node = Node(short(fm.group('sym')), count)
        # Walk up until we find an open ancestor shallower than this line, and
        # that can still contain it (inclusive counts must dominate children:
        # `sample` appends a flat "leaf subtree" summary that must not be
        # attached under the last thread).
        while len(stack) > 1 and (
                stack[-1]._depth >= depth or stack[-1].count < count):
            stack.pop()
        node._depth = depth
        parent = stack[-1]
        # Reject an attachment that would over-count the parent: `sample`
        # appends a flat "leaf subtree" summary after the last thread, and it
        # renders at a column that would otherwise nest under that thread.
        if sum(c.count for c in parent.children) + count > parent.count:
            continue
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


def build_svg(self_stacks, out_path, title, width=1700):
    tree = defaultdict(lambda: defaultdict(int))

    def insert(stack, n):
        node = ()
        for frame in stack:
            tree[node][frame] += n
            node = node + (frame,)

    for stack, n in self_stacks.items():
        insert(stack, n)

    total = sum(self_stacks.values())
    row_h = 15
    max_depth = max((len(s) for s in self_stacks), default=1)
    height = (max_depth + 2) * row_h + 46
    px_per = (width - 24) / max(total, 1)

    def esc(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    rects = []

    def walk(node, x, y):
        children = tree.get(node)
        if not children:
            return
        cx = x
        for child, n in sorted(children.items(), key=lambda kv: -kv[1]):
            w = n * px_per
            if w < 0.7:
                cx += w
                continue
            frac = n / total
            hue = (sum(child.encode()) * 41) % 360
            light = 78 - min(32, int(frac * 120))
            rects.append(
                f'<rect x="{cx:.2f}" y="{y}" width="{w:.2f}" height="{row_h - 1}"'
                f' fill="hsl({hue},62%,{light}%)"><title>{esc(child)}&'
                f'#10;{n} samples ({100.0 * frac:.2f}%)</title></rect>')
            if w > 40:
                label = child if len(child) < 74 else child[:71] + '…'
                rects.append(
                    f'<text x="{cx + 3:.2f}" y="{y + row_h - 4}">'
                    f'{esc(label)}</text>')
            walk(node + (child,), cx, y + row_h)
            cx += w

    walk((), 12, 34)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}">',
        '<style>text{font-family:Menlo,monospace;font-size:9px;fill:#111;'
        'pointer-events:none}.t{font-family:-apple-system,sans-serif;'
        'font-size:13px;font-weight:600}</style>',
        '<rect width="100%" height="100%" fill="#fcfcfc"/>',
        f'<text class="t" x="12" y="20">{esc(title)}</text>',
        f'<text x="12" y="30" style="font-size:10px">'
        f'{total} samples of self time · width = time · depth = stack</text>',
    ] + rects + ['</svg>']
    open(out_path, 'w').write('\n'.join(svg))
    return total, max_depth


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    roots = parse(src)
    self_stacks = defaultdict(int)
    problems = []
    for r in roots:
        collect(r, (), self_stacks, problems)
    total, depth = build_svg(
        self_stacks, dst,
        'av2enc --nightly-b 24 f QP119 --cpu-threads 8 (sample, 1 ms)')
    print(f'{len(roots)} threads, {len(self_stacks)} stacks, '
          f'{total} self samples, depth {depth}, '
          f'{len(problems)} over-counted nodes')
    for p in problems[:5]:
        print('  OVERCOUNT', p)
