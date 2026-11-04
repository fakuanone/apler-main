import math


def minimum_cost_bipartite_assignment(cost_matrix):
    """Minimum-cost maximum-cardinality matching of either rectangular orientation."""
    if hasattr(cost_matrix, "detach"):
        costs = cost_matrix.detach().float().cpu().tolist()
    else:
        costs = [[float(value) for value in row] for row in cost_matrix]
    prediction_count = len(costs)
    target_count = len(costs[0]) if prediction_count else 0
    if any(len(row) != target_count for row in costs):
        raise ValueError("cost_matrix must be rectangular.")
    if any(not math.isfinite(value) for row in costs for value in row):
        raise ValueError("cost_matrix values must all be finite.")
    if prediction_count == 0 or target_count == 0:
        return []
    if prediction_count < target_count:
        transposed = [list(column) for column in zip(*costs)]
        return sorted(((target, prediction) for prediction, target in
                       minimum_cost_bipartite_assignment(transposed)), key=lambda pair: pair[1])

    matrix = [
        [costs[prediction_index][target_index] for prediction_index in range(prediction_count)]
        for target_index in range(target_count)
    ]
    u = [0.0] * (target_count + 1)
    v = [0.0] * (prediction_count + 1)
    p = [0] * (prediction_count + 1)
    way = [0] * (prediction_count + 1)
    infinity = float("inf")

    for target_index in range(1, target_count + 1):
        p[0] = target_index
        current_column = 0
        min_values = [infinity] * (prediction_count + 1)
        used = [False] * (prediction_count + 1)
        while True:
            used[current_column] = True
            current_row = p[current_column]
            delta = infinity
            next_column = 0
            for prediction_index in range(1, prediction_count + 1):
                if used[prediction_index]:
                    continue
                reduced_cost = (
                    matrix[current_row - 1][prediction_index - 1]
                    - u[current_row]
                    - v[prediction_index]
                )
                if reduced_cost < min_values[prediction_index]:
                    min_values[prediction_index] = reduced_cost
                    way[prediction_index] = current_column
                if min_values[prediction_index] < delta:
                    delta = min_values[prediction_index]
                    next_column = prediction_index
            for prediction_index in range(prediction_count + 1):
                if used[prediction_index]:
                    u[p[prediction_index]] += delta
                    v[prediction_index] -= delta
                else:
                    min_values[prediction_index] -= delta
            current_column = next_column
            if p[current_column] == 0:
                break
        while True:
            previous_column = way[current_column]
            p[current_column] = p[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    pairs = []
    for prediction_index in range(1, prediction_count + 1):
        if p[prediction_index] != 0:
            pairs.append((prediction_index - 1, p[prediction_index] - 1))
    return sorted(pairs, key=lambda pair: pair[1])
