
from lxml import etree

ns = '{http://ProjectMalmo.microsoft.com}'


def put(elem, path, text, attrib=None):
    """Put text into given path of element and sub-elements (where path separated by a '.').
    Optional arg attrib can be used to select the element's attribute by that name.
    Somewhat modelled on C++ PropertyTree's put."""
    return get_or_put(elem, path, text, attrib)


def get(elem, path, attrib=None):
    """Get text from given path of element and sub-elements (where path separated by a '.').
    Optional arg attrib can be used to select the element's attribute by that name.
    Somewhat modelled on C++ PropertyTree's put."""
    return get_or_put(elem, path, None, attrib)


def get_or_put(elem, path, text=None, attrib=None):
    """Get or put text by given path of element and sub-elements (where path separated by a '.').
    Get if text arg is None else put.
    Second optional arg attrib can be used to get from or put into an attribute by attribute name.
    """
    e = elem
    path2 = path.split('.')
    p = path2.pop(0)
    if e.tag != ns + p:
        raise Exception('Have ' + e.tag + ' expected first element to be: ' + p)
    for p in path2:
        e2 = e.find(ns + p)
        if e2 is None:
            e2 = etree.Element(ns + p)
            e.append(e2)
        e = e2
    if text:
        if attrib is None:
            e.text = str(text)
        else:
            e.attrib[attrib] = text
        return None
    else:
        if attrib is None:
            return e.text
        else:
            return e.attrib[attrib]
