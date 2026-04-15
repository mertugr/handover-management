"""
Central configuration file.
All constants are defined here. Keep this file identical across implementations
to ensure reproducible results.
"""

# General
RANDOM_SEED = 42

# Cell grid
GRID_ROWS   = 3
GRID_COLS   = 3
GRID_WIDTH  = 3000.0   # metres
GRID_HEIGHT = 3000.0   # metres

# Mobility (Random Waypoint model)
MIN_SPEED_MPS = 1.0    # m/s
MAX_SPEED_MPS = 30.0   # m/s
TIME_STEP_S   = 1.0    # seconds per simulation tick
PAUSE_PROB    = 0.10   # probability of pausing at a waypoint
MIN_PAUSE_S   = 1.0    # seconds
MAX_PAUSE_S   = 30.0   # seconds
DIR_NOISE_STD = 0.15   # radians (~8.6 degrees)
MARGIN        = 50.0   # metres from grid boundary

# RSSI / Physical layer (3GPP COST-231 Urban Macro)
P_TX_DBM      = 43.0   # eNB transmit power (dBm)
CABLE_LOSS_DB =  2.0   # feeder losses (dB)
ANTENNA_GAIN  = 15.0   # eNB antenna gain (dBi)
PL_CONST      = 128.1  # path-loss intercept at d = 1 km (dB)
PL_SLOPE      =  37.6  # path-loss slope (dB/decade of km)
SHADOWING_STD =   8.0  # log-normal shadowing std (dB)
MIN_RSSI      = -120.0 # dBm floor
MAX_RSSI      =  -30.0 # dBm ceiling
MIN_DISTANCE  =  10.0  # metres, avoids log(0)

# Dataset generation
NUM_USERS  = 60   # number of simulated users
NUM_STEPS  = 700  # time steps recorded per user
BURN_IN    = 100  # warm-up steps discarded at the start
LOOKAHEAD  = 5    # steps ahead for the next_cell label

# Random Forest hyperparameters
RF_N_ESTIMATORS     = 100
RF_MAX_DEPTH        = 20
RF_MIN_SAMPLES_LEAF = 4

# Threshold (A3-event) controller
THR_HYSTERESIS_DB = 3.0  # dB margin required to trigger handover
THR_TTT_STEPS     = 3    # consecutive steps condition must hold (TTT)

# ML controller
ML_CONFIDENCE_THRESHOLD = 0.55  # minimum prediction confidence to trigger HO
ML_COOLDOWN_STEPS       = 8     # steps blocked after a handover
ML_MIN_GAIN_DB          = 2.0   # minimum RSSI gain required (dB)
ML_BOUNCEBACK_WINDOW    = 15    # steps before returning to previous cell is allowed

# Train / val / test split
SPLIT_TEST_SIZE = 0.20
SPLIT_VAL_SIZE  = 0.10
