from ._base import Vanilla
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
from .Teacher import Teacher
from .Teacher_lamb import Teacher_lamb
from .EKD import EKD
from .KD4eModel import KD4eModel
from .DKD4eModel import DKD4eModel
from .MLKD4eModel import MLKD4eModel
from .Monitor import Monitor
from .EKDIN import EKDIN
from .CAT_KD import CAT_KD
from .EKD_WKD_F import EKDWKD
from .EKD_SimKD import EKDSimKD

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
    "Teacher": Teacher,
    "Teacher_lamb": Teacher_lamb,
    "EKD": EKD,
    "KD4eModel": KD4eModel,
    "DKD4eModel": DKD4eModel,
    "MLKD4eModel": MLKD4eModel,
    "Monitor": Monitor,
    "EKDIN": EKDIN,
    "CAT_KD": CAT_KD,
    "EKDWKD": EKDWKD,
    "EKDSimKD": EKDSimKD,
}
