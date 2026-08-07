"""Utility functions for the to-do app."""

import json
import os


def export_todos(todos, filepath="todos_export.txt"):
    """Export to-dos to a plain text file."""
    f = open(filepath, "w")
    for i in range(len(todos)):
        todo = todos[i]
        status = "DONE" if todo["done"] == True else "TODO"
        line = "%d. [%s] %s\n" % (todo["id"], status, todo["title"])
        f.write(line)
    f.close()
    print("Exported %d to-dos to %s" % (len(todos), filepath))


def filter_by_priority(todos, priority, include_done=True):
    """Filter to-dos by priority level."""
    results = []
    for i in range(len(todos)):
        if todos[i]["priority"] == priority:
            if include_done == True or todos[i]["done"] == False:
                results.append(todos[i])
    return results


def get_stats(todos):
    """Get summary statistics for to-dos."""
    total = len(todos)
    done = 0
    for i in range(len(todos)):
        if todos[i]["done"] == True:
            done = done + 1
    pending = total - done

    priorities = {}
    for i in range(len(todos)):
        p = todos[i]["priority"]
        if p in priorities:
            priorities[p] = priorities[p] + 1
        else:
            priorities[p] = 1

    print("Total: %d | Done: %d | Pending: %d" % (total, done, pending))
    for p in priorities:
        print("  %s: %d" % (p, priorities[p]))
