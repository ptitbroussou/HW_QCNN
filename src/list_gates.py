from src.toolbox import PQNN_building_brick, QCNN_RBS_based_VQC_bottom_channel


def full_connection_circuit(n):
    """ n=5
    0: ─╭B─╭B─╭B─╭B─╭S──────────╭S──────────╭S──────────╭S──────────┤
    1: ─╰S─│──│──│──╰B─╭B─╭B─╭B─│──╭S───────│──╭S───────│──╭S───────┤
    2: ────╰S─│──│─────╰S─│──│──╰B─╰B─╭B─╭B─│──│──╭S────│──│──╭S────┤
    3: ───────╰S─│────────╰S─│────────╰S─│──╰B─╰B─╰B─╭B─│──│──│──╭S─┤
    4: ──────────╰S──────────╰S──────────╰S──────────╰S─╰B─╰B─╰B─╰B─┤
    """
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def half_connection_circuit(n):
    """ n=5
    ─╭B─╭B────╭B───────╭B──────────┤
    ─╰S─│──╭B─│──╭B────│──╭B───────┤
    ────╰S─╰S─│──│──╭B─│──│──╭B────┤
    ──────────╰S─╰S─╰S─│──│──│──╭B─┤
    ───────────────────╰S─╰S─╰S─╰S─┤
    """
    return [(i, j) for i in range(n) for j in range(n) if i > j]


def full_reverse_connection_circuit(n):
    """ n=5
    0: ──────────╭S──────────╭S──────────╭S──────────╭S─╭B─╭B─╭B─╭B─┤
    1: ───────╭S─│────────╭S─│────────╭S─│──╭B─╭B─╭B─╰B─│──│──│──╰S─┤
    2: ────╭S─│──│─────╭S─│──│──╭B─╭B─╰B─╰B─│──│──╰S────│──│──╰S────┤
    3: ─╭S─│──│──│──╭B─╰B─╰B─╰B─│──╰S───────│──╰S───────│──╰S───────┤
    4: ─╰B─╰B─╰B─╰B─╰S──────────╰S──────────╰S──────────╰S──────────┤
    """
    return [(i, j) for i in range(n - 1, -1, -1) for j in range(n - 1, -1, -1) if i != j]


def slide_circuit(n):
    """ n=8
    ─╭B───────────────────╭S─┤
    ─╰S─╭B────────────────│──┤
    ────╰S─╭B─────────────│──┤
    ───────╰S─╭B──────────│──┤
    ──────────╰S─╭B───────│──┤
    ─────────────╰S─╭B────│──┤
    ────────────────╰S─╭B─│──┤
    ───────────────────╰S─╰B─┤
    """
    return [(i, (i + 1) % n) for i in range(n)]


def full_pyramid_circuit(n):
    """ n=8
    0: ─╭B────╭B────╭B────╭B────╭B────╭B────╭B─┤
    1: ─╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B─╰S─┤
    2: ────╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B─╰S────┤
    3: ───────╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B─╰S───────┤
    4: ──────────╰S─╭B─╰S─╭B─╰S─╭B─╰S──────────┤
    5: ─────────────╰S─╭B─╰S─╭B─╰S─────────────┤
    6: ────────────────╰S─╭B─╰S────────────────┤
    7: ───────────────────╰S───────────────────┤
    """
    list_gates = []
    _, PQNN_dictionary, _ = PQNN_building_brick(0, n, index_first_RBS=0, index_first_param=0)
    for key in PQNN_dictionary:
        list_gates.append(PQNN_dictionary[key])
    return list_gates


def get_reduced_layers_structure(n, out):
    """ n=8
    0: ─╭B────╭B────╭B────╭B──────────┤
    1: ─╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B───────┤
    2: ────╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B────┤
    3: ───────╰S─╭B─╰S─╭B─╰S─╭B─╰S─╭B─┤
    4: ──────────╰S─╭B─╰S─╭B─╰S─╭B─╰S─┤
    5: ─────────────╰S─╭B─╰S─╭B─╰S────┤
    6: ────────────────╰S─╭B─╰S───────┤
    7: ───────────────────╰S──────────┤
    """
    list_gates = []
    PQNN_param_dictionary, PQNN_dictionary, PQNN_layer = PQNN_building_brick(0, n, index_first_RBS=0,
                                                                             index_first_param=0)
    for key in PQNN_dictionary:
        list_gates.append(PQNN_dictionary[key])
    list_gates.reverse()

    list_gates_delete = []
    PQNN_param_dictionary, PQNN_dictionary, PQNN_layer = PQNN_building_brick(0, n - out, index_first_RBS=0,
                                                                             index_first_param=0)
    for key in PQNN_dictionary:
        list_gates.append(PQNN_dictionary[key])

    for e in list_gates_delete:
        list_gates.remove(e)
    list_gates.reverse()
    return list_gates


def drip_circuit(n):
    """ n=8
    0: ─╭B──────────╭B───────╭B────╭B───────────────────┤
    1: ─│──╭B───────│──╭B────│──╭B─╰S─╭B────────────────┤
    2: ─│──│──╭B────│──│──╭B─╰S─│──╭B─╰S─╭B─────────────┤
    3: ─│──│──│──╭B─╰S─│──│──╭B─╰S─│──╭B─╰S─╭B──────────┤
    4: ─╰S─│──│──│─────╰S─│──│──╭B─╰S─│──╭B─╰S─╭B───────┤
    5: ────╰S─│──│────────╰S─│──│─────╰S─│──╭B─╰S─╭B────┤
    6: ───────╰S─│───────────╰S─│────────╰S─│─────╰S─╭B─┤
    7: ──────────╰S─────────────╰S──────────╰S───────╰S─┤
    """
    list_gates = []
    for k in range(n // 2, 0, -1):
        i = 0
        while i + k < n:
            list_gates.append((i, i + k))
            i += 1
    return list_gates


def butterfly_circuit(n):
    """ n=8
    0: ─╭B───────╭B────╭B────┤
    1: ─│──╭B────│──╭B─╰S────┤
    2: ─│──│──╭B─╰S─│──╭B────┤
    3: ─│──│──│──╭B─╰S─╰S────┤
    4: ─╰S─│──│──│──╭B────╭B─┤
    5: ────╰S─│──│──│──╭B─╰S─┤
    6: ───────╰S─│──╰S─│──╭B─┤
    7: ──────────╰S────╰S─╰S─┤
    """
    gates = []
    stage = n // 2
    while stage > 0:
        for i in range(0, n, stage * 2):
            for j in range(stage):
                if i + j + stage < n:
                    gates.append((i + j, i + j + stage))
        stage //= 2
    return gates


def X_circuit(n):
    """ n=8
    0: ─╭B────────────────╭B─┤
    1: ─╰S─╭B──────────╭B─╰S─┤
    2: ────╰S─╭B────╭B─╰S────┤
    3: ───────╰S─╭B─╰S───────┤
    4: ───────╭B─╰S─╭B───────┤
    5: ────╭B─╰S────╰S─╭B────┤
    6: ─╭B─╰S──────────╰S─╭B─┤
    7: ─╰S────────────────╰S─┤
    """
    list = []
    for i in range(n // 2 - 1):
        list.append((i, i + 1))
        list.append((n - i - 2, n - i - 1))
    list.append((n // 2 - 1, n // 2))
    list2 = []
    for i in range(n // 2 - 1):
        list2.append((i, i + 1))
        list2.append((n - i - 2, n - i - 1))
    return list + list2[::-1]


def conv_3D_gates(I, K, J):
    """
    I=4, K=4, J=4
     0: ─╭B────╭B────╭B─┤
     1: ─╰S─╭B─╰S─╭B─╰S─┤
     2: ────╰S─╭B─╰S────┤
     3: ───────╰S───────┤
     4: ─╭B────╭B────╭B─┤
     5: ─╰S─╭B─╰S─╭B─╰S─┤
     6: ────╰S─╭B─╰S────┤
     7: ───────╰S───────┤
     8: ─╭B────╭B────╭B─┤
     9: ─╰S─╭B─╰S─╭B─╰S─┤
    10: ────╰S─╭B─╰S────┤
    11: ───────╰S───────┤
    """
    list_gates = []
    _, Param_dictionary, RBS_dictionary = QCNN_RBS_based_VQC_bottom_channel(I, K, J)
    for key in RBS_dictionary:
        list_gates.append((RBS_dictionary[key], RBS_dictionary[key] + 1))
    return list_gates
