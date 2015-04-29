#!/usr/bin/python
import sys
from nltk.tree import Tree

parse_file = sys.argv[1]
line_num = int(sys.argv[2])
fout = open('tree.tex','w')
print >>fout,r'''\documentclass[tikz]{standalone}
\usepackage{CJKutf8}
\usepackage{color}
\usepackage{tikz}
\usepackage{tikz-qtree}
\usetikzlibrary{calc}
\thispagestyle{empty}
\begin{document}
\begin{CJK}{UTF8}{gbsn}

\begin{tikzpicture}'''
f = open(parse_file)
for i,line in enumerate(f):
	if i == line_num:
		line = line.split(' ||||| ')
		s = line[0].replace('$','\$').replace('|||','$|||$').replace('(_','<_').replace(')_','>_')
		ws = line[1].split()
		tree = Tree.parse(s)
		h = tree.height()
		print >>fout,r'\begin{scope}'
		leaf_pos = tree.treepositions('leaves')
		tree[leaf_pos[0]] = r'\edge; {' + tree[leaf_pos[0]] + '}'
		tree[leaf_pos[-1]] = r'\edge; {' + tree[leaf_pos[-1]] + '}'
		idx = 0
		for line in tree.pprint_latex_qtree().split('\n'):
			if ';' in line:
				line = line.replace('{','\\node(n{}) {{'.format(idx)).replace('}','};').replace('%','\%')
				idx += 1
			print >>fout,line.replace('_','\ ').replace('<','(').replace('>',')')
		print >>fout,'\draw (n0 |- 0,{}pt) node (s0) {{{}}};'.format(-h*28-10,ws[0])
		print >>fout,'\draw (n1 |- 0,{}pt) node (s1) {{{}}};'.format(-h*28-10,ws[-1])
		print >>fout,'\draw (n0 |- 0,{}pt) node (p0) {{{}}};'.format(-h*28-30,0)
		print >>fout,'\draw (n1 |- 0,{}pt) node (p1) {{{}}};'.format(-h*28-30,len(ws)-1)
		for i in range(1,len(ws)-1):
			print >>fout,'\\node () at ($(s0)!{}!(s1)$) {{{}}};'.format(float(i)/(len(ws)-1),ws[i])
			print >>fout,'\\node () at ($(p0)!{}!(p1)$) {{{}}};'.format(float(i)/(len(ws)-1),i)
		break
print >>fout,r'''
\end{scope}
\end{tikzpicture}
\end{CJK}
\end{document}
'''
f.close()
