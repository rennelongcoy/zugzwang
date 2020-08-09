import numpy as np

# Chess parameters
CHESS_FILE_COUNT = 8
CHESS_RANK_COUNT = 8

# Board Image parameters
ONE_SQUARE_SIZE = 50
EIGHTH_FILE = (CHESS_FILE_COUNT - 1) * ONE_SQUARE_SIZE + 1
EIGHTH_RANK = (CHESS_RANK_COUNT - 1) * ONE_SQUARE_SIZE + 1

# Board State parameters
INITIAL_BOARD_STATE = [[ 1,  1,  1,  1,  1,  1,  1,  1], # Rank A
                       [ 1,  1,  1,  1,  1,  1,  1,  1], # Rank B
                       [ 0,  0,  0,  0,  0,  0,  0,  0], # Rank C
                       [ 0,  0,  0,  0,  0,  0,  0,  0], # Rank D
                       [ 0,  0,  0,  0,  0,  0,  0,  0], # Rank E
                       [ 0,  0,  0,  0,  0,  0,  0,  0], # Rank F
                       [-1, -1, -1, -1, -1, -1, -1, -1], # Rank G
                       [-1, -1, -1, -1, -1, -1, -1, -1]] # Rank H

ZERO_BOARD_STATE = [[ 0, 0, 0, 0, 0, 0, 0, 0], # Rank A
                    [ 0, 0, 0, 0, 0, 0, 0, 0], # Rank B
                    [ 0, 0, 0, 0, 0, 0, 0, 0], # Rank C
                    [ 0, 0, 0, 0, 0, 0, 0, 0], # Rank D
                    [ 0, 0, 0, 0, 0, 0, 0, 0], # Rank E
                    [ 0, 0, 0, 0, 0, 0, 0, 0], # Rank F
                    [ 0, 0, 0, 0, 0, 0, 0, 0], # Rank G
                    [ 0, 0, 0, 0, 0, 0, 0, 0]] # Rank H

# Region of Interest parameters
ROI_SIZE        = 400
ROI_LEFT_SIDE   = 120
ROI_TOP_SIDE    = 40
ROI_RIGHT_SIDE  = ROI_LEFT_SIDE + ROI_SIZE
ROI_BOTTOM_SIDE = ROI_TOP_SIDE + ROI_SIZE

# Overlay parameters
OVERLAY_TOP_LEFT_X         = ROI_LEFT_SIDE
OVERLAY_TOP_LEFT_Y         = ROI_TOP_SIDE
OVERLAY_BOTTOM_RIGHT_X     = ROI_RIGHT_SIDE - 1
OVERLAY_BOTTOM_RIGHT_Y     = ROI_BOTTOM_SIDE - 1
OVERLAY_TOP_LEFT_POINT     = (OVERLAY_TOP_LEFT_X, OVERLAY_TOP_LEFT_Y)
OVERLAY_BOTTOM_RIGHT_POINT = (OVERLAY_BOTTOM_RIGHT_X, OVERLAY_BOTTOM_RIGHT_Y)
OVERLAY_COLOR_BGR          = (0, 0, 255)
OVERLAY_THICKNESS          = 1