DIR             = 'D2T'

WEBNLG          = "https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en/train"

EPOCHS          = 4
LR              = 1e-3
EPS             = (1e-30, 1e-3),
CLIP_THRESHOLD  = 1.0,
DECAY_RATE      = -0.8,
BETA1           = None,
WEIGHT_DECAY    = 0.0,
RELATIVE_STEP   = False,
SCALE_PARAMETER = False,
WARMUP_INIT     = False

BATCH_SIZE      = 8
NUM_OF_EPOCHS   = 1