"""
Example config that sets lamda_1_list as a Python list of lists (incremental ELLA per task).
Use with: python scripts_t5/ella/run_long_sequence.py --order 5 --run_number 1 --config_file scripts_t5/ella/example_config_lamda_1_list.py
"""

# One list per task (15 tasks for order 5/6/4). Each inner list: 2 = first/last half, 3 = first/mid/last third.
lamda_1_list = [
    [0, 0],       # task 1: no ELLA
    [0.1, 10.0],  # task 2
    [0.1, 10.0],  # task 3
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.1, 10.0],
]
