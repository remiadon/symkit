from sympy import symbols


def symbol_generator():
    i = 0
    while True:  # TODO : set a limit
        yield symbols(f"__symkit_{i}")
        i += 1
