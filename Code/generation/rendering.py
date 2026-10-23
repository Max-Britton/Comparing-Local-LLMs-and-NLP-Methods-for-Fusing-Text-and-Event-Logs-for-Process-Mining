from __future__ import annotations


import pandas as pd


ERROR_CLASSES = (
    "Air Pressure Error", "Lubrication Deficiency", "Power Outage",
    "Part Misalignment", "Clamping Error", "Control Software Glitch",
    "Cooling System Failure", "Feed Path Error",
)


STYLES = (
    "Keep it short, four or five sentences.",
    "Take your time over it; it can run long.",
    "Keep it plain and matter-of-fact.",
    "Write it the way you speak, not the way a form asks.",
    "Write it in a hurry, part of it in fragments.",
    "Write it carefully and in full sentences.",
    "Tell it in the order things happened.",
    "Start from where you ended up, then how you got there.",
)

CLOSING = (
    "Write one paragraph of plain text and nothing else. Every detail listed above must stay "
    "recoverable from what you write."
)


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}
_MONTHS = ("January February March April May June July August September October November "
           "December").split()
_ORDINALS = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh "
             "twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth "
             "nineteenth twentieth").split()


def _words(number: int) -> str:
    if number < 20:
        return _ONES[number]
    tens, ones = divmod(number, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def _ordinal(day: int) -> str:
    if day <= 20:
        return _ORDINALS[day - 1]
    if day == 30:
        return "thirtieth"
    if day % 10 == 0:
        return _TENS[day // 10] + "ieth"
    return _TENS[day // 10] + "-" + _ORDINALS[day % 10 - 1]


def _part_of_day(hour: int) -> str:
    if hour < 12:
        return "in the morning"
    if hour < 17:
        return "in the afternoon"
    if hour < 21:
        return "in the evening"
    return "at night"


def render_when(stamp: pd.Timestamp, form: str) -> str:
    hour12 = stamp.hour % 12 or 12
    if form == "24h":
        return f"{stamp:%H:%M} on {stamp.day} {_MONTHS[stamp.month - 1]} {stamp.year}"
    if form == "12h":
        marker = "am" if stamp.hour < 12 else "pm"
        return (f"{hour12}:{stamp:%M} {marker} on {stamp.day} "
                f"{_MONTHS[stamp.month - 1]} {stamp.year}")
    minute = stamp.minute
    spoken = ("o'clock" if minute == 0 else
              f"oh {_words(minute)}" if minute < 10 else _words(minute))
    return (f"{_words(hour12)} {spoken} {_part_of_day(stamp.hour)}, "
            f"{_ordinal(stamp.day)} of {_MONTHS[stamp.month - 1]} {stamp.year}")
OFFSETS = (7, 9, 11, 14, 18, 23, 26, 31)


