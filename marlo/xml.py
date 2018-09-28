
from lxml import etree

ns = '{http://ProjectMalmo.microsoft.com}'


def put(elem, path, text, attrib=None):
    """put text into given path of element and sub-elements (where path separated by a '.').
    Somewhat modelled on C++ PropertyTree's put."""
    e = elem
    path2 = path.split('.')
    p = path2.pop(0)
    if e.tag != ns + p:
        raise Exception('Have ' + e.tag + ' expected first element to be: ' + p)
    for p in path2:
        print(p)
        e2 = e.find(ns + p)
        if e2 is None:
            e2 = etree.Element(ns + p)
            e.append(e2)
        e = e2
    if attrib is None:
        e.text = str(text)
    else:
        e.attrib[attrib] = text

