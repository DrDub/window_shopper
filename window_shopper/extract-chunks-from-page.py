from bs4 import BeautifulSoup, Tag, NavigableString
import codecs

import sys
#sys.setrecursionlimit(2500)

import collections

inline_tags = set([ "tt", "i", "b", "u", "s", "strike", "big", "small", "em", "string",
                "dfn", "code", "samp", "kbd", "var", "cite", "acronym", "a", "img",
                "applet", "object", "font", "basefont", "script", "map", "q", "sub",
                "sup", "span", "bdo", "iframe", "input", "select", "textarea", "label",
                "button" ])

def extract_para(node, f):
    nodes = collections.deque(node.contents)
    #nodes = collections.deque()
    #for c in node.contents:
    #    nodes.append(c)
    while len(nodes) > 0:
        c = nodes.popleft()
        if type(c) == Tag:
            if c.name in inline_tags:
                pass
            else:
                f.write(u'\n')
            if not (c.name == 'script' or c.name == 'style'):
                f.write(u' ')
                for cc in reversed(c.contents):
                    nodes.appendleft(cc)
        elif type(c) == NavigableString:
            f.write(c.string.replace('\n', ' '))
        else:
            f.write(u' ')
            if 'contents' in dir(c):
                for cc in c.contents:
                    nodes.append(cc)


try:
    soup = BeautifulSoup(open(sys.argv[1]))
    output_title = codecs.open(sys.argv[3], encoding='utf-8', mode='w')
    head = soup.find('head')
    if head:
        title_tag= head.find('title')
        if title_tag:
            output_title.write(title_tag.string.replace('\n', ' '))
    output_title.close()

    
    output = codecs.open(sys.argv[2], encoding='utf-8', mode='w')
    extract_para(soup, output)
    output.close()
except Exception, exc:
    print exc

