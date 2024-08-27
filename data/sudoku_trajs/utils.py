board_width = 9

def actionTupleToAction(action_tuple):
        '''
        Converts (i,j,digit) tuple to an action integer in [0, board_width^3]
        '''
        (i,j,digit) = action_tuple

        return i * (board_width**2) + j * (board_width) + digit - 1


def actionToActionTuple(action_num:int):
    '''
    Given an action num in [0, board_width^3), convert to a tuple which represents
        (i,j, digit) where i,j in [0,board_width) and digit in [1,board_width]
    '''
    i = action_num // (board_width**2)
    remainder = action_num % (board_width**2)
    j = remainder // board_width 
    digit = (remainder % board_width) + 1

    return (i, j, digit)       