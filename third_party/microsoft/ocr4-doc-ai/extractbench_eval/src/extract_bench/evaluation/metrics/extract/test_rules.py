"""Test rule implementations for extract evaluation."""

from typing import Any

from extract_bench.evaluation.metrics.extract.test_types import ExtractTestType


def _resolve_path(data: dict[str, Any] | list[Any], path: str) -> Any | None:
    """
    Resolve a dot-notation path in the data structure.

    Uses simplified dot-notation format:
    - Empty string "" refers to the root (entire data structure)
    - Nested paths use dots: "items.conditions"
    - Array indices are numeric segments: "items.0.name"

    :param data: The data structure to navigate (dict or list)
    :param path: Dot-notation path (e.g., "", "general_conditions", "items.0.conditions")
    :return: The value at the path, or None if path doesn't exist
    """
    # Handle root path (empty string)
    if path == "":
        return data

    # Split path into segments by dot
    segments = path.split(".")

    current: Any = data
    for segment in segments:
        if current is None:
            return None

        # Try to access as dict key
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
        # Try to access as list index
        elif isinstance(current, list):
            try:
                index = int(segment)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            except ValueError:
                return None
        else:
            # Can't navigate further
            return None

    return current


class ExtractTestRule:
    """Base class for extract test rules."""

    def __init__(self, rule_data: dict[str, Any]):
        """
        Initialize a test rule from a dictionary.

        :param rule_data: Dictionary containing rule definition
        """
        self.type = rule_data.get("type")
        self.description = rule_data.get("description")
        self.name = rule_data.get("name")

    def run(self, extracted_data: dict[str, Any] | list[Any]) -> tuple[bool, str]:
        """
        Run the test rule against extracted data.

        :param extracted_data: Extracted JSON data to test (dict or list)
        :return: Tuple of (passed, explanation)
        """
        raise NotImplementedError("Subclasses must implement run()")


class ArrayLengthRule(ExtractTestRule):
    """Test rule for validating array length at a JSON path."""

    def __init__(self, rule_data: dict[str, Any]):
        """
        Initialize an array length rule.

        :param rule_data: Dictionary containing:
            - type: "array_length"
            - path: Dot-notation path to the array (required, "" for root)
            - operator: Comparison operator (required)
            - value: Expected length (number or string, required)
            - description: Optional description
            - name: Optional rule name
        """
        super().__init__(rule_data)

        # Validate required fields (path can be empty string for root)
        path = rule_data.get("path")
        if path is None:
            raise ValueError("ArrayLengthRule requires 'path' field")
        self.path: str = path

        operator = rule_data.get("operator")
        if not operator:
            raise ValueError("ArrayLengthRule requires 'operator' field")
        self.operator: str = operator

        value = rule_data.get("value")
        if value is None:
            raise ValueError("ArrayLengthRule requires 'value' field")
        self.value: int | float | str = value

        # Convert value to int
        try:
            if isinstance(self.value, str):
                self.expected_length = int(self.value)
            elif isinstance(self.value, (int, float)):
                self.expected_length = int(self.value)
            else:
                raise ValueError(f"Value must be convertible to integer: {self.value}")
        except (ValueError, TypeError) as e:
            msg = f"Invalid value: '{self.value}' (must be convertible to integer)"
            raise ValueError(msg) from e

        if self.expected_length < 0:
            raise ValueError(f"Value must be non-negative: {self.expected_length}")

        # Validate operator
        valid_operators = {
            "equals",
            "greater_than",
            "less_than",
            "greater_than_or_equal",
            "less_than_or_equal",
            # Aliases for convenience
            "eq",
            "gt",
            "lt",
            "gte",
            "lte",
        }
        if self.operator not in valid_operators:
            valid_ops_str = ", ".join(sorted(valid_operators))
            raise ValueError(f"Invalid operator: '{self.operator}'. Must be one of: {valid_ops_str}")

    def run(self, extracted_data: dict[str, Any] | list[Any]) -> tuple[bool, str]:
        """
        Run the array length rule against extracted data.

        :param extracted_data: Extracted JSON data to test (dict or list)
        :return: Tuple of (passed, explanation)
        """
        # Resolve path
        value_at_path = _resolve_path(extracted_data, self.path)

        if value_at_path is None:
            path_display = "root" if self.path == "" else f"'{self.path}'"
            rule_id = f"'{self.name}'" if self.name else f"at {path_display}"
            return False, f"Path {path_display} not found in extracted data"

        # Check if value is an array
        if not isinstance(value_at_path, list):
            actual_type = type(value_at_path).__name__
            path_display = "root" if self.path == "" else f"'{self.path}'"
            rule_id = f"'{self.name}'" if self.name else f"at {path_display}"
            return False, f"Value {rule_id} is not an array (found type: {actual_type})"

        # Get actual length
        actual_length = len(value_at_path)

        # Normalize operator (handle aliases)
        operator_map = {
            "eq": "equals",
            "gt": "greater_than",
            "lt": "less_than",
            "gte": "greater_than_or_equal",
            "lte": "less_than_or_equal",
        }
        normalized_operator = operator_map.get(self.operator, self.operator)

        # Perform comparison
        passed = False
        if normalized_operator == "equals":
            passed = actual_length == self.expected_length
        elif normalized_operator == "greater_than":
            passed = actual_length > self.expected_length
        elif normalized_operator == "less_than":
            passed = actual_length < self.expected_length
        elif normalized_operator == "greater_than_or_equal":
            passed = actual_length >= self.expected_length
        elif normalized_operator == "less_than_or_equal":
            passed = actual_length <= self.expected_length

        # Generate explanation
        path_display = "root" if self.path == "" else f"'{self.path}'"
        rule_id = f"'{self.name}'" if self.name else f"at {path_display}"
        if passed:
            explanation = (
                f"Array {rule_id} has length {actual_length}, "
                f"which {normalized_operator.replace('_', ' ')} {self.expected_length}"
            )
        else:
            explanation = (
                f"Array {rule_id} has length {actual_length}, "
                f"expected {normalized_operator.replace('_', ' ')} {self.expected_length}"
            )

        # Include description if available
        if self.description:
            explanation = f"{self.description}: {explanation}"

        return passed, explanation


class ArrayHeadRule(ExtractTestRule):
    """Test rule for validating the first N elements of an array."""

    def __init__(self, rule_data: dict[str, Any]):
        """
        Initialize an array head rule.

        :param rule_data: Dictionary containing:
            - type: "array_head"
            - path: Dot-notation path to the array (required, "" for root)
            - count: Number of elements to check from the start (required)
            - expected: List of expected values for the head elements (required)
            - description: Optional description
            - name: Optional rule name
        """
        super().__init__(rule_data)

        # Validate required fields (path can be empty string for root)
        path = rule_data.get("path")
        if path is None:
            raise ValueError("ArrayHeadRule requires 'path' field")
        self.path: str = path

        count = rule_data.get("count")
        if count is None:
            raise ValueError("ArrayHeadRule requires 'count' field")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"ArrayHeadRule 'count' must be a positive integer: {count}")
        self.count: int = count

        expected = rule_data.get("expected")
        if expected is None:
            raise ValueError("ArrayHeadRule requires 'expected' field")
        if not isinstance(expected, list):
            raise ValueError("ArrayHeadRule 'expected' must be a list")
        if len(expected) != count:
            raise ValueError(f"ArrayHeadRule 'expected' length ({len(expected)}) must match 'count' ({count})")
        self.expected: list[Any] = expected

    def run(self, extracted_data: dict[str, Any] | list[Any]) -> tuple[bool, str]:
        """
        Run the array head rule against extracted data.

        :param extracted_data: Extracted JSON data to test (dict or list)
        :return: Tuple of (passed, explanation)
        """
        # Resolve path
        value_at_path = _resolve_path(extracted_data, self.path)
        path_display = "root" if self.path == "" else f"'{self.path}'"
        rule_id = f"'{self.name}'" if self.name else f"at {path_display}"

        if value_at_path is None:
            return False, f"Path {path_display} not found in extracted data"

        # Check if value is an array
        if not isinstance(value_at_path, list):
            actual_type = type(value_at_path).__name__
            return False, f"Value {rule_id} is not an array (found type: {actual_type})"

        # Check if array has enough elements
        if len(value_at_path) < self.count:
            return False, (f"Array {rule_id} has only {len(value_at_path)} elements, expected at least {self.count}")

        # Compare head elements
        actual_head = value_at_path[: self.count]
        if actual_head == self.expected:
            explanation = f"Array {rule_id} head ({self.count} elements) matches expected values"
            if self.description:
                explanation = f"{self.description}: {explanation}"
            return True, explanation

        # Find first mismatch for better error message
        for i, (actual, expected) in enumerate(zip(actual_head, self.expected, strict=True)):
            if actual != expected:
                explanation = f"Array {rule_id} head mismatch at index {i}: expected {expected!r}, got {actual!r}"
                if self.description:
                    explanation = f"{self.description}: {explanation}"
                return False, explanation

        # Should not reach here, but just in case
        explanation = f"Array {rule_id} head does not match expected values"
        if self.description:
            explanation = f"{self.description}: {explanation}"
        return False, explanation


class ArrayTailRule(ExtractTestRule):
    """Test rule for validating the last N elements of an array."""

    def __init__(self, rule_data: dict[str, Any]):
        """
        Initialize an array tail rule.

        :param rule_data: Dictionary containing:
            - type: "array_tail"
            - path: Dot-notation path to the array (required, "" for root)
            - count: Number of elements to check from the end (required)
            - expected: List of expected values for the tail elements (required)
            - description: Optional description
            - name: Optional rule name
        """
        super().__init__(rule_data)

        # Validate required fields (path can be empty string for root)
        path = rule_data.get("path")
        if path is None:
            raise ValueError("ArrayTailRule requires 'path' field")
        self.path: str = path

        count = rule_data.get("count")
        if count is None:
            raise ValueError("ArrayTailRule requires 'count' field")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"ArrayTailRule 'count' must be a positive integer: {count}")
        self.count: int = count

        expected = rule_data.get("expected")
        if expected is None:
            raise ValueError("ArrayTailRule requires 'expected' field")
        if not isinstance(expected, list):
            raise ValueError("ArrayTailRule 'expected' must be a list")
        if len(expected) != count:
            raise ValueError(f"ArrayTailRule 'expected' length ({len(expected)}) must match 'count' ({count})")
        self.expected: list[Any] = expected

    def run(self, extracted_data: dict[str, Any] | list[Any]) -> tuple[bool, str]:
        """
        Run the array tail rule against extracted data.

        :param extracted_data: Extracted JSON data to test (dict or list)
        :return: Tuple of (passed, explanation)
        """
        # Resolve path
        value_at_path = _resolve_path(extracted_data, self.path)
        path_display = "root" if self.path == "" else f"'{self.path}'"
        rule_id = f"'{self.name}'" if self.name else f"at {path_display}"

        if value_at_path is None:
            return False, f"Path {path_display} not found in extracted data"

        # Check if value is an array
        if not isinstance(value_at_path, list):
            actual_type = type(value_at_path).__name__
            return False, f"Value {rule_id} is not an array (found type: {actual_type})"

        # Check if array has enough elements
        if len(value_at_path) < self.count:
            return False, (f"Array {rule_id} has only {len(value_at_path)} elements, expected at least {self.count}")

        # Compare tail elements
        actual_tail = value_at_path[-self.count :]
        if actual_tail == self.expected:
            explanation = f"Array {rule_id} tail ({self.count} elements) matches expected values"
            if self.description:
                explanation = f"{self.description}: {explanation}"
            return True, explanation

        # Find first mismatch for better error message
        for i, (actual, expected) in enumerate(zip(actual_tail, self.expected, strict=True)):
            if actual != expected:
                # Calculate actual index in the original array
                actual_index = len(value_at_path) - self.count + i
                explanation = (
                    f"Array {rule_id} tail mismatch at index {actual_index} "
                    f"(tail position {i}): expected {expected!r}, got {actual!r}"
                )
                if self.description:
                    explanation = f"{self.description}: {explanation}"
                return False, explanation

        # Should not reach here, but just in case
        explanation = f"Array {rule_id} tail does not match expected values"
        if self.description:
            explanation = f"{self.description}: {explanation}"
        return False, explanation


def create_test_rule(rule_data: dict[str, Any]) -> ExtractTestRule:
    """
    Create a test rule from a dictionary.

    :param rule_data: Dictionary containing rule definition
    :return: ExtractTestRule instance
    :raises ValueError: If rule type is unknown or invalid
    """
    rule_type = rule_data.get("type")
    if not rule_type:
        raise ValueError("Rule must have a 'type' field")

    if rule_type == ExtractTestType.ARRAY_LENGTH.value:
        return ArrayLengthRule(rule_data)
    elif rule_type == ExtractTestType.ARRAY_HEAD.value:
        return ArrayHeadRule(rule_data)
    elif rule_type == ExtractTestType.ARRAY_TAIL.value:
        return ArrayTailRule(rule_data)
    else:
        raise ValueError(f"Unknown test type: {rule_type}")
