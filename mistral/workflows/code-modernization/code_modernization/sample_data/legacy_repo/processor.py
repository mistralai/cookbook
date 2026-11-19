# Python 2 data processor: xrange, raw_input, except syntax, has_key.
# Each construct has a direct Python 3 replacement the LLM should recognise.

import sys


def process_batch(items, batch_size=10):
    results = []
    for i in xrange(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        results.append(_process_chunk(chunk))
    return results


def _process_chunk(chunk):
    processed = []
    for item in chunk:
        if not isinstance(item, dict):
            continue
        if item.has_key("value"):
            processed.append(item["value"] * 2)
    return processed


def interactive_run():
    try:
        limit = int(raw_input("Enter batch limit: "))
    except ValueError:
        print "Invalid input, using default"
        limit = 100

    data = [{"value": i} for i in xrange(limit)]
    result = process_batch(data)
    print "Processed %d batches" % len(result)
    return result


def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError, e:
        print "Division error: %s" % e
        return None


if __name__ == "__main__":
    interactive_run()
