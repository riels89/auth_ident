"""
Module for managing nested dictionary collections.
"""

# nestdict.py by Adam Szieberth (2013)
# Python 3.3+

class NestedDict(dict):
    """
    Class for managing nested dictionary structures. Normally, it works
    like a builtin dictionary. However, if it gets a list as an argument,
    it will iterate through that list assuming all elements of that list
    as a key for the subdirectory chain.

    NestedDict implements module level functions and makes managing nested
    dictionary structure easier.

    Instead of having a complicated way to manage extending or
    overwriting, NestedDict has a lock property (not decorated!) which
    allows or prohibits all alterations on the particular NestedDict
    instance. Warning! If you do not pass  list (even if it has only one
    element) to __setitem__, the superclass' method will be used which
    sets the item regardless of lock state! 

    If you want more sophisticated behavior than full access/prohibition,
    you can still use module level functions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = False

    def __getitem__(self, *args):
        if isinstance(args[0], list):
            return getitem(self, args[0])
        return super().__getitem__(*args)

    def __setitem__(self, *args):
        if isinstance(args[0], list):
            lock = self.get_lock(args[0])
            if not lock:
                return setitem(self, args[0], args[1],
                               overwrite=not lock, restruct=not lock,
                               dict_type=type(self))
            else:
                return False
        else:
            super().__setitem__(*args)
            return True

    def get_lock(self, path):
        """
        Returns the state of lock on the given path. In fact it walks on
        the path as long as possible, and returns the state of the last
        lock it can get. 
        """
        lock = self.lock
        level = 1
        while level <= len(path):
            try:
                lock = getitem(self, path[:level]).lock
            except (KeyError, AttributeError):
                break
            level += 1
        return lock

    def func_if_unlocked(self, *args):
        """
        The default func_if_unlocked function for self.merge() method
        which checks for lock on a path and returns True if path is
        unlocked.
        """ 
        path = args[0]
        return not self.get_lock(path)

    def lock_close(self, recursively=True):
        """
        Locks locks.
        """
        self.lock = True
        if recursively:
            for p in self.paths(of_values=False):
                self.__getitem__(p).lock = True

    def lock_open(self, recursively=True):
        """
        Unlocks locks.
        """
        self.lock = False
        if recursively:
            for p in self.paths(of_values=False):
                self.__getitem__(p).lock = False

    def merge(self, *dictobjs, restruct=True):
        """
        Same as module level function merge. It needs less arguments
        though since it uses self.func_if_unlocked() method to manage
        extend and overwrite permissions.
        """ 
        merge(self, *dictobjs,
              func_if_extend=self.func_if_unlocked,
              func_if_overwrite=self.func_if_unlocked,
              restruct=restruct,
              dict_type=type(self))

    def paths(self, of_values=True):
        """
        Same as module level function paths.
        """
        return paths(self, of_values=of_values)

def getitem(dictobj, path):
    """
    Returns the element of a nested dictionary structure which is on the
    given path. 
    """
    _validate_path(path)
    if len(path) == 1:
        return dictobj[path[0]]
    else:
        return getitem(dictobj[path[0]], path[1:])

def setitem(dictobj, path, value, overwrite=True, restruct=True,
        dict_type=dict):
    """
    Sets a dictionary item on a given path to a given value.
      - Returns True if value on path has been set.
      - Returns False if there was a value on the given path which was not
        overwritten by the function.
      - Returns None if there was a value on the given path which was
        identical to value.

    If restruct=True then when a value blocks the path, that value get
    cleared by an empty dictionary to make way forward.
    """
    _validate_path(path)

    try:
        one_step = dictobj[path[0]]
    except KeyError:
        if len(path) == 1:
            dictobj[path[0]] = value
            return True
        else:
            dictobj[path[0]] = dict_type()
            one_step = dictobj[path[0]]
    else:
        if len(path) == 1 and one_step == value:
            return None
        elif len(path) == 1 and overwrite is False:
            return False
        elif len(path) == 1 and overwrite is True:
            dictobj[path[0]] = value
            return True
        else:
            if not isinstance(one_step, dict):
                if overwrite is True and restruct is True: ##TEST
                    dictobj[path[0]] = dict_type()
                    one_step = dictobj[path[0]]
                else:
                    return False
    return setitem(one_step, path[1:], value, overwrite=overwrite,
                restruct=restruct, dict_type=dict_type)

def paths(dictobj, of_values=True, past_keys=[]):
    """
    Generator to iterate through branches. Used by merge function, but
    can be useful for other object management stuffs.

    By default it returns paths of values. However, if of_values=False
    then it returns the paths of all subdirectories.
    """
    for key in dictobj.keys():
        path = past_keys + [key]
        if not isinstance(dictobj[key], dict):
            if of_values is True:
                yield path
        else:
            if of_values is False:
                yield path
            yield from paths(dictobj[key], of_values=of_values,
                             past_keys=path)

def merge(*dictobjs,
          func_if_extend=True,
          func_if_overwrite=True,
          restruct=True,
          dict_type=dict,
          return_new=False):
    """
    Merges one dictionary with one or more another.

    By default it mutates the first dictobj. However, if return_new=True
    then it returns a new dictionary object typed recursively to
    dict_type. If you want no retypeing, use copy.deepcopy(), and pass the
    copied dictionary as first argument.

    To make mergeing more flexible, you are able to control how extension
    overwriting should be done (both are allowed by default). By setting
    func_if_overwrite to False, overwriting becomes disabled. By setting
    func_if_extend to False, extension becomes disabled and you can only
    update existing values if overwriting is enabled. If both are
    disabled, no alteration will be made, so this scenario makes no sense,
    but allowed.

    Moreover you can pass functions to the two mentioned arguments which
    will be called with the path (list of keys), dictobj1, dictobj2
    arguments and expected to return True or False.
    """
    if return_new is True:
        d = retype(dictobjs[0], dict_type)
    elif return_new is False:
        d = dictobjs[0]

    for dictobj in dictobjs[1:]:
        for p in paths(dictobj):
            try:
                getitem(d, p)
            except KeyError:
                    if hasattr(func_if_extend, '__call__'):
                        ex = func_if_extend(p, d, dictobj)
                    else:
                        ex = func_if_extend
                    if ex:
                        setitem(d, p, getitem(dictobj, p),
                                dict_type=dict_type)
            else:
                if getitem(d, p) != getitem(dictobj, p):
                    if hasattr(func_if_overwrite, '__call__'):
                        ow = func_if_overwrite(p, d, dictobj)
                    else:
                        ow = func_if_overwrite
                    restruct_ = restruct and ow 
                    setitem(d, p, getitem(dictobj, p),
                            overwrite=ow,
                            restruct=restruct_,
                            dict_type=dict_type)
    return d

def retype(dictobj, dict_type):
    """
    Recursively modifies the type of a dictionary object and returns a new
    dictionary of type dict_type. You can also use this function instead
    of copy.deepcopy() for dictionaries.
    """
    def walker(dictobj):
        for k in dictobj.keys():
            if isinstance(dictobj[k], dict):
                yield (k, dict_type(walker(dictobj[k])))
            else:
                yield (k, dictobj[k])
    d = dict_type(walker(dictobj))
    return d


def _validate_path(path):
    if not isinstance(path, list):
        raise TypeError('path argument have to be a list')
    if not path:
        raise Exception('path argument have to be a nonempty list')


def main():
    import pprint
    print('nestdict.py by Adam Szieberth')
    print(__doc__)
    print('Example for Stack Overflow question #635483:\n')
    inp_data =[(['new jersey', 'mercer county', 'plumbers'], 3),
               (['new jersey', 'mercer county', 'programmers'], 81),
               (['new jersey', 'middlesex county', 'programmers'], 81),
               (['new jersey', 'middlesex county', 'salesmen'], 62),
               (['new york', 'queens county', 'plumbers'], 9),
               (['new york', 'queens county', 'salesmen'], 36)]
    print('Input data:\n')
    pprint.PrettyPrinter(indent=1).pprint(inp_data)
    print('\n>>> data = NestedDict()')
    data = NestedDict()
    print('>>> for d in inp_data:')
    print('>>>     data[d[0]] = d[1]\n')
    for d in inp_data:
        data[d[0]] = d[1]
    print('Result:\n')
    pprint.PrettyPrinter(indent=0).pprint(data)
    return data

if __name__ == '__main__':
    data = main()
