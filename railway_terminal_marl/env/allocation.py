from __future__ import annotations

from typing import List, Optional, Tuple


def maximum_weight_matching(
    weights: List[List[Optional[float]]],
) -> List[Tuple[int, int]]:
    """Return a maximum-weight one-to-one assignment for feasible edges."""
    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    if not row_count or not column_count:
        return []
    if any(len(row) != column_count for row in weights):
        raise ValueError("weights must be a rectangular matrix")

    size = max(row_count, column_count)
    feasible = [value for row in weights for value in row if value is not None]
    if not feasible:
        return []
    largest_weight = max(feasible)
    forbidden_cost = (abs(largest_weight) + 1.0) * (size + 1) * 1_000_000
    costs = []
    for row_index in range(size):
        row_costs = []
        for column_index in range(size):
            if row_index >= row_count or column_index >= column_count:
                row_costs.append(largest_weight)
            else:
                weight = weights[row_index][column_index]
                row_costs.append(largest_weight - weight if weight is not None else forbidden_cost)
        costs.append(row_costs)

    # Hungarian algorithm for a square minimum-cost assignment.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matched_row[0] = row
        column0 = 0
        min_cost = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                reduced_cost = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if reduced_cost < min_cost[column]:
                    min_cost[column] = reduced_cost
                    predecessor[column] = column0
                if min_cost[column] < delta:
                    delta = min_cost[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    min_cost[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    result = []
    for column in range(1, size + 1):
        row = matched_row[column] - 1
        col = column - 1
        if row < row_count and col < column_count and weights[row][col] is not None:
            result.append((row, col))
    return result
