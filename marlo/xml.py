
from lxml import etree

ns = '{http://ProjectMalmo.microsoft.com}'


def put(elem, path, text, attrib=None):
    """Put text into given path of element and sub-elements (where path separated by a '.').
    Optional arg attrib can be used to select the element's attribute by that name for put.
    Somewhat modelled on C++ PropertyTree's put."""
    return get_or_put(elem, path, text, attrib=attrib)


def get(elem, path, attrib=None):
    """Get text from given path of element and sub-elements (where path separated by a '.').
    Optional arg attrib can be used to select the element's attribute by that name to get.
    """
    return get_or_put(elem, path, None, attrib=attrib)


def get_or_put(elem, path, text=None, attrib=None):
    """Get or put text by given path of element and sub-elements (where path separated by a '.').
    Get if text arg is None else put the text value into given path.
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


def get_sub_element(elem, path):
    """Get a sub element by path or None if not found."""
    e = elem
    path2 = path.split('.')
    p = path2.pop(0)
    if e.tag != ns + p:
        raise Exception('Have ' + e.tag + ' expected first element to be: ' + p)
    for p in path2:
        e = e.find(ns + p)
        if e is None:
            return None
    return e


def put_all(elem, path, sub_path, text=None, attrib=None):
    e = elem
    path2 = path.split('.')
    p = path2.pop(0)
    p_last = path2.pop()
    if e.tag != ns + p:
        raise Exception('Have ' + e.tag + ' expected first element to be: ' + p)
    for p in path2:
        e2 = e.find(ns + p)
        if e2 is None:
            e2 = etree.Element(ns + p)
            e.append(e2)
        e = e2
    children = e.findall(ns + p_last)
    if children is None:
        e2 = etree.Element(ns + p)
        e.append(e2)
        children = [e2]
    for child in children:
        put(child, p_last + '.' + sub_path, text, attrib)
