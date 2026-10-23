VERSION = "manufacturing-symptoms-10"
ERROR_CLASSES = (
    "Air Pressure Error",
    "Lubrication Deficiency",
    "Power Outage",
    "Part Misalignment",
    "Clamping Error",
    "Control Software Glitch",
    "Cooling System Failure",
    "Feed Path Error",
    "Sensor Malfunction",
    "Material Defect",
)
ALIASES = {"Software Glitch": "Control Software Glitch"}
CLASSIFICATION_LABELS = tuple(
    "Software Glitch" if name == "Control Software Glitch" else name for name in ERROR_CLASSES
) + ("Other",)

DEFINITIONS = {
    "Air Pressure Error": (
        "The underlying problem is insufficient or unstable compressed-air delivery to the operation. "
        "Changes in the pneumatic action and the work must fit this cause."
    ),
    "Lubrication Deficiency": (
        "The underlying problem is inadequate lubrication at moving contacts. "
        "The observations must fit impaired friction reduction, not merely an unexplained rise in process temperature."
    ),
    "Power Outage": (
        "The underlying problem is an interruption of electrical supply to the affected equipment. "
        "A disruption of work must fit that loss of supply rather than merely a commanded pause."
    ),
    "Part Misalignment": (
        "The underlying problem is incorrect relative alignment of the workpiece and processing geometry. "
        "Incorrect geometry is distinct from a part that loses its restraint and moves under the process load."
    ),
    "Clamping Error": (
        "The underlying problem is inadequate restraint of the workpiece during the operation. "
        "The observations must fit loss of holding, not merely a securely held part in the wrong initial alignment."
    ),
    "Control Software Glitch": (
        "The underlying problem is incorrect handling of commands, sequence or state by the control software. "
        "The relationship between intended operation and actual sequence must fit this cause."
    ),
    "Cooling System Failure": (
        "The underlying problem is inadequate removal of process heat by the cooling system. "
        "The observations must fit impaired heat removal, not merely resistance at an inadequately lubricated moving contact."
    ),
    "Feed Path Error": (
        "The underlying problem is an incorrect, interrupted or obstructed path of advancement through the operation. "
        "The observations must fit the physical progression of tool or material, not merely a generic bad finish."
    ),
    "Sensor Malfunction": (
        "The fault concerns the measurement or detection signal itself, rather than "
        "the control program or an actual change of the measured process condition."
    ),
    "Material Defect": (
        "The fault concerns a defect already present in the workpiece material, "
        "rather than incorrect positioning, inadequate holding, or damage caused "
        "by a different machine fault."
    ),
}
