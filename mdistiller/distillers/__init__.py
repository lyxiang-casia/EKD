from ._base import Vanilla, Vanilla_feat2logit
from .KD import KD
from .MLKD import MLKD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .Sonly import Sonly
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .Evidential_Teacher import Teacher
from .EKD import EKD
from .CAT_KD import CAT_KD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "MLKD": MLKD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "Sonly": Sonly,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "Evidential_Teacher": Evidential_Teacher,
    "EKD": EKD,
    "Monitor": Monitor,
    "CAT_KD": CAT_KD,
}
