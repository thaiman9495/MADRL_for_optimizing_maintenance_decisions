
def structure_function(s):
    """
    This function describes the structure of a multi-component system

    Args:
        s: component failure status vector,
         s[i] = 0 -> component[i] is failed at inspection time,
         s[i] = 1 -> component[i] is functioning at inspection time
    Returns: system faliure indicator
    """
    sub_1 = s[0]
    sub_2 = 1 - (1 - s[1]) * (1 - s[2])
    sub_3 = 1 - (1 - s[3]) * (1 - s[4]) * (1 - s[5])
    sub_4 = 1 - (1 - s[6]) * (1 - s[7])
    sub_5 = 1 - (1 - s[8]) * (1 - s[9]) * (1 - s[10])
    return sub_1 * sub_2 * sub_3 * sub_4 * sub_5

