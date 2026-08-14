"""A simple CLI to-do list manager."""

import json
import os
import sys


TODO_FILE = os.path.join(os.path.dirname(__file__), "todos.json")


def load_todos(filepath=TODO_FILE):
    """Load todos from a JSON file."""
    if os.path.exists(filepath) == True:
        f = open(filepath, "r")
        try:
            data = json.loads(f.read())
        except:
            data = []
        f.close()
        return data
    else:
        return []


def save_todos(todos, filepath=TODO_FILE):
    """Save todos to a JSON file."""
    f = open(filepath, "w")
    f.write(json.dumps(todos, indent=2))
    f.close()


def add_todo(todos, title, priority="medium"):
    """Add a new to-do item."""
    if type(title) != str or title == "":
        print("Error: title must be a non-empty string")
        return todos
    new_todo = {
        "id": len(todos) + 1,
        "title": title,
        "priority": priority,
        "done": False,
    }
    todos.append(new_todo)
    print("Added: %s (priority: %s)" % (title, priority))
    return todos


def complete_todo(todos, todo_id):
    """Mark a to-do item as complete."""
    for i in range(len(todos)):
        if todos[i]["id"] == todo_id:
            todos[i]["done"] = True
            print("Completed: %s" % todos[i]["title"])
            return todos
    print("Error: to-do #%d not found" % todo_id)
    return todos


def list_todos(todos, show_done=True):
    """Display all to-do items."""
    if len(todos) == 0:
        print("No to-dos found.")
        return
    for i in range(len(todos)):
        todo = todos[i]
        if show_done == False and todo["done"] == True:
            continue
        status = "x" if todo["done"] == True else " "
        print("[%s] #%d: %s (priority: %s)" % (status, todo["id"], todo["title"], todo["priority"]))


def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <command> [args]")
        print("Commands: add, complete, list")
        sys.exit(1)

    command = sys.argv[1]
    todos = load_todos()

    if command == "add":
        if len(sys.argv) < 3:
            print("Usage: python app.py add <title> [priority]")
            sys.exit(1)
        title = sys.argv[2]
        priority = sys.argv[3] if len(sys.argv) > 3 else "medium"
        todos = add_todo(todos, title, priority)
        save_todos(todos)
    elif command == "complete":
        if len(sys.argv) < 3:
            print("Usage: python app.py complete <id>")
            sys.exit(1)
        todo_id = int(sys.argv[2])
        todos = complete_todo(todos, todo_id)
        save_todos(todos)
    elif command == "list":
        show_done = True
        if len(sys.argv) > 2 and sys.argv[2] == "--active":
            show_done = False
        list_todos(todos, show_done)
    else:
        print("Unknown command: %s" % command)
        sys.exit(1)


if __name__ == "__main__":
    main()
