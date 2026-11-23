# Python 2 utility helpers: print statement, iteritems, unicode casts.
# This file is syntactically valid Python 2 and intentionally uses
# deprecated idioms so the LLM has concrete changes to make.

import sys


def format_record(record):
    items = []
    for key, value in record.iteritems():
        items.append(u"%s=%s" % (unicode(key), unicode(value)))
    return u", ".join(items)


def log(message, level="INFO"):
    print "[%s] %s" % (level, message)


def merge_dicts(base, overrides):
    result = dict(base)
    for key, value in overrides.iteritems():
        result[key] = value
    return result


def to_unicode_list(items):
    return [unicode(item) for item in items]


if __name__ == "__main__":
    sample = {"name": "Alice", "role": "admin"}
    print format_record(sample)
    log("utility module loaded")
