#!/usr/bin/env python3
"""Build an INTERACTIVE flame graph (HTML) from a macOS `sample` call-graph dump.

Why HTML and not a bare .svg: a `.svg` opened as an image (how a preview pane
renders it) runs no script and shows no `<title>` tooltips, so the graph is
inert.  This emits one self-contained `.html` with inline SVG plus a little
script, giving:

  * hover highlight + a details panel (inclusive %, self %, samples),
  * click to zoom into a frame (click the background to zoom out),
  * a search box that highlights matching frames,
  * reset via the title.

`sample` renders a call tree with a line drawing whose characters its man page
calls arbitrary, so the parser uses the *column of the count* as the level and
derives self time as inclusive - sum(children).  It also rejects an attachment
that would over-count its parent, because `sample` appends a flat "leaf
subtree" summary after the last thread.  A correct parse reports 0
over-counted nodes, which is printed.
"""
import argparse
import json
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
    # 2-character cell per level; the chars themselves are arbitrary.
    return (len(indent) - 6) // 2


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
        depth = marker_depth(m.group('indent'))
        count = int(m.group('count'))
        node = Node(short(fm.group('sym')), count)
        while len(stack) > 1 and (stack[-1]._depth >= depth or
                                  stack[-1].count < count):
            stack.pop()
        parent = stack[-1]
        # Reject an attachment that would over-count the parent (sample's
        # trailing flat "leaf subtree" summary).
        if sum(c.count for c in parent.children) + count > parent.count:
            continue
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


def build_html(self_stacks, out_path, title, width=1700):
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
        return (s.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;').replace('"', '&quot;'))

    rects = []

    # self time keyed by full path, for the details panel
    self_by_path = {}
    for st, n in self_stacks.items():
        self_by_path[st] = self_by_path.get(st, 0) + n

    def walk(node, x, y, path):
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
            full_path = path + [child]
            rects.append(
                f'<rect class="f" x="{cx:.2f}" y="{y}" width="{w:.2f}" '
                f'height="{row_h - 1}" fill="hsl({hue},62%,{light}%)" '
                f'data-name="{esc(child)}" data-samples="{n}" '
                f'data-pct="{100.0 * frac:.3f}" '
                f'data-self="{self_by_path.get(tuple(full_path), 0)}" '
                f'data-path="{esc(" > ".join(full_path))}"/>')
            if w > 40:
                label = child if len(child) < 74 else child[:71] + '…'
                rects.append(
                    f'<text x="{cx + 3:.2f}" y="{y + row_h - 4}" '
                    f'pointer-events="none">{esc(label)}</text>')
            walk(node + (child,), cx, y + row_h, full_path)
            cx += w

    walk((), 12, 34, [])
    svg = [
        f'<svg id="g" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect id="bg" x="0" y="0" width="100%" height="100%" fill="#fcfcfc"/>',
        f'<text x="12" y="20" style="font:600 13px -apple-system,sans-serif">'
        f'{esc(title)}</text>',
        f'<text x="12" y="30" style="font:10px -apple-system,sans-serif;'
        f'fill:#555">{total} samples of self time · width = time · '
        f'depth = stack</text>',
    ] + rects + ['</svg>']

    html = f"""<!DOCTYPE html>
<meta charset="utf-8">
<title>{esc(title)}</title>
<style>
 html,body{{margin:0;background:#fff;font:13px -apple-system,sans-serif}}
 #bar{{position:sticky;top:0;display:flex;gap:12px;align-items:center;
      padding:8px 12px;background:#f4f4f6;border-bottom:1px solid #ddd;z-index:2}}
 #info{{color:#333;font-family:Menlo,monospace;font-size:12px;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
 input{{font:inherit;padding:4px 8px;width:240px}}
 #wrap{{overflow:auto;height:calc(100vh - 46px)}}
 rect.f:hover{{stroke:#000;stroke-width:1}}
 rect.hit{{stroke:#d00;stroke-width:1.5}}
</style>
<div id="bar">
  <input id="q" placeholder="search frames…" spellcheck="false">
  <span id="info">hover a frame · click to zoom · click the background to reset</span>
</div>
<div id="wrap">
{chr(10).join(svg)}
</div>
<script>
const g = document.getElementById('g');
const wrap = document.getElementById('wrap');
const info = document.getElementById('info');
let zoom = null;
const frames = [...g.querySelectorAll('rect.f')];

function show(el) {{
  const pct = (+el.dataset.pct).toFixed(2);
  const self = +el.dataset.self;
  const sel = frames.length || 1;
  info.textContent = el.dataset.name + '  ·  ' + pct + '% inclusive (' +
    el.dataset.samples + ' samples)  ·  self ' + self + ' (' +
    (100 * self / sel).toFixed(2) + '%)';
}}
frames.forEach(el => {{
  el.addEventListener('mouseenter', () => show(el));
  el.addEventListener('click', ev => {{
    ev.stopPropagation();
    if (zoom === el) {{ reset(); return; }}
    zoom = el;
    const bx = +el.getAttribute('x'), by = +el.getAttribute('y');
    const bw = +el.getAttribute('width'), bh = +el.getAttribute('height');
    const pad = 20;
    g.setAttribute('viewBox', (bx - pad) + ' ' + (by - pad) + ' ' +
      (bw + 2 * pad) + ' ' + (bh + 2 * pad));
    g.setAttribute('width', Math.max(600, bw + 2 * pad));
    g.setAttribute('height', bh + 2 * pad);
  }});
}});
function reset() {{
  zoom = null;
  g.setAttribute('viewBox', '0 0 ' + {width} + ' ' + {height});
  g.setAttribute('width', {width});
  g.setAttribute('height', {height});
  info.textContent = 'hover a frame · click to zoom · click the background to reset';
}}
document.getElementById('bg').addEventListener('click', reset);
document.getElementById('q').addEventListener('input', ev => {{
  const t = ev.target.value.toLowerCase();
  frames.forEach(el => {{
    const m = t && el.dataset.name.toLowerCase().includes(t);
    el.classList.toggle('hit', !!m);
  }});
}});
</script>
"""
    open(out_path, 'w').write(html)
    return total, max_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sample')
    ap.add_argument('out_html')
    ap.add_argument('--title', default='flame graph')
    ap.add_argument('--width', type=int, default=1700)
    a = ap.parse_args()
    roots = parse(a.sample)
    stacks = defaultdict(int)
    problems = []
    for r in roots:
        collect(r, (), stacks, problems)
    total, depth = build_html(stacks, a.out_html, a.title, a.width)
    print(f'{len(roots)} threads, {len(stacks)} stacks, {total} self samples, '
          f'depth {depth}, {len(problems)} over-counted nodes')


if __name__ == '__main__':
    main()
