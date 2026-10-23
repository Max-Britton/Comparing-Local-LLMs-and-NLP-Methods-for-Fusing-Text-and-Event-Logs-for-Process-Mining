from .rendering import ERROR_CLASSES as ORIGINAL_CLASSES

VERSION = 'manufacturing-symptoms-10-v1'
NEW_CLASSES = ('Sensor Malfunction', 'Material Defect')
ERROR_CLASSES = ORIGINAL_CLASSES + NEW_CLASSES

DEFINITIONS = {
    'Sensor Malfunction': (
        'The fault concerns the measurement or detection signal itself, rather than '
        'the control program or an actual change of the measured process condition.'
    ),
    'Material Defect': (
        'The fault concerns a defect already present in the workpiece material, '
        'rather than incorrect positioning, inadequate holding, or damage caused '
        'by a different machine fault.'
    ),
}
