"""
Phonetic Encoding Algorithms for Business Entity Resolution.
Provides Double Metaphone encoding with zero external C-dependencies.
If `metaphone` or `jellyfish` is installed, it uses them; otherwise, it uses
the built-in pure-Python Double Metaphone implementation.
"""

import re
from typing import Tuple

try:
    from metaphone import doublemetaphone as _dm_ext
    def double_metaphone(token: str) -> Tuple[str, str]:
        res = _dm_ext(token)
        return (res[0] or "", res[1] or "")
except ImportError:
    try:
        from jellyfish import metaphone as _jm_ext
        def double_metaphone(token: str) -> Tuple[str, str]:
            res = _jm_ext(token)
            return (res or "", res or "")
    except ImportError:
        _dm_ext = None


if '_dm_ext' not in locals() or _dm_ext is None:
    # Pure Python Double Metaphone Implementation (Philips' algorithm)
    def double_metaphone(token: str) -> Tuple[str, str]:
        """
        Pure-Python Double Metaphone implementation.
        Returns a tuple of (primary_code, secondary_code).
        """
        if not token:
            return ("", "")

        original = token.upper()
        # Clean non-alpha
        s = re.sub(r"[^A-Z]", "", original)
        if not s:
            return ("", "")

        length = len(s)
        current = 0
        primary = []
        secondary = []

        # Pad for safety
        def char_at(idx: int) -> str:
            if 0 <= idx < length:
                return s[idx]
            return ""

        def substr(start: int, length_: int) -> str:
            if start < 0 or start >= length:
                return ""
            return s[start : start + length_]

        # Initial checks
        if substr(0, 2) in ("GN", "KN", "PN", "WR", "PS"):
            current += 1
        elif char_at(0) == "X":
            primary.append("S")
            secondary.append("S")
            current += 1

        while current < length:
            ch = char_at(current)

            if ch in ("A", "E", "I", "O", "U", "Y"):
                if current == 0:
                    primary.append("A")
                    secondary.append("A")
                current += 1
            elif ch == "B":
                primary.append("P")
                secondary.append("P")
                if char_at(current + 1) == "B":
                    current += 2
                else:
                    current += 1
            elif ch == "C":
                # Various C rules
                if current > 1 and not char_at(current - 2) in ("A", "E", "I", "O", "U", "Y") and substr(current - 1, 3) == "ACH" and (char_at(current + 2) not in ("I", "E") or substr(current - 2, 6) in ("BACHER", "MACHER")):
                    primary.append("K")
                    secondary.append("K")
                    current += 2
                elif current == 0 and substr(current, 6) == "CAESAR":
                    primary.append("S")
                    secondary.append("S")
                    current += 2
                elif substr(current, 4) == "CHIA":
                    primary.append("K")
                    secondary.append("K")
                    current += 2
                elif substr(current, 2) == "CH":
                    if current > 0 and substr(current, 4) == "CHAE":
                        primary.append("K")
                        secondary.append("X")
                        current += 2
                    elif current == 0 and (substr(current + 1, 5) in ("HARAC", "HARIS") or substr(current + 1, 3) in ("HOR", "HYM", "HIA", "HEM")):
                        primary.append("K")
                        secondary.append("K")
                        current += 2
                    else:
                        if current == 0:
                            primary.append("X")
                            secondary.append("X")
                        else:
                            primary.append("K")
                            secondary.append("X")
                        current += 2
                elif substr(current, 2) == "CZ":
                    primary.append("S")
                    secondary.append("X")
                    current += 2
                elif substr(current, 3) == "CIA":
                    primary.append("X")
                    secondary.append("X")
                    current += 3
                elif substr(current, 2) == "CC" and not (current == 1 and char_at(0) == "M"):
                    if char_at(current + 2) in ("I", "E", "H") and substr(current + 2, 2) != "HU":
                        if (current == 1 and char_at(current - 1) == "A") or substr(current - 1, 5) in ("UCCEE", "UCCES"):
                            primary.append("KS")
                            secondary.append("KS")
                        else:
                            primary.append("X")
                            secondary.append("X")
                        current += 3
                    else:
                        primary.append("K")
                        secondary.append("K")
                        current += 2
                elif substr(current, 2) in ("CK", "CG", "CQ"):
                    primary.append("K")
                    secondary.append("K")
                    current += 2
                elif substr(current, 2) in ("CI", "CE", "CY"):
                    primary.append("S")
                    secondary.append("S")
                    current += 2
                else:
                    primary.append("K")
                    secondary.append("K")
                    if substr(current + 1, 2) in (" C", " Q", " G"):
                        current += 3
                    elif char_at(current + 1) in ("C", "K", "Q") and substr(current + 1, 2) not in ("CE", "CI"):
                        current += 2
                    else:
                        current += 1
            elif ch == "D":
                if substr(current, 2) == "DG":
                    if char_at(current + 2) in ("I", "E", "Y"):
                        primary.append("J")
                        secondary.append("J")
                        current += 3
                    else:
                        primary.append("TK")
                        secondary.append("TK")
                        current += 2
                elif substr(current, 2) in ("DT", "DD"):
                    primary.append("T")
                    secondary.append("T")
                    current += 2
                else:
                    primary.append("T")
                    secondary.append("T")
                    current += 1
            elif ch == "F":
                primary.append("F")
                secondary.append("F")
                if char_at(current + 1) == "F":
                    current += 2
                else:
                    current += 1
            elif ch == "G":
                if char_at(current + 1) == "H":
                    if current > 0 and not char_at(current - 1) in ("A", "E", "I", "O", "U", "Y"):
                        primary.append("K")
                        secondary.append("K")
                        current += 2
                    elif current == 0:
                        if char_at(current + 2) == "I":
                            primary.append("J")
                            secondary.append("J")
                        else:
                            primary.append("K")
                            secondary.append("K")
                        current += 2
                    elif current > 1 and char_at(current - 2) in ("B", "H", "D"):
                        current += 2
                    else:
                        primary.append("K")
                        secondary.append("K")
                        current += 2
                elif char_at(current + 1) == "N":
                    if current == 1 and char_at(0) in ("A", "E", "I", "O", "U", "Y") and not (substr(current + 2, 2) == "EY" or char_at(current + 2) == "Y"):
                        primary.append("KN")
                        secondary.append("N")
                    else:
                        primary.append("N")
                        secondary.append("KN")
                    current += 2
                elif substr(current, 2) == "GG":
                    primary.append("K")
                    secondary.append("K")
                    current += 2
                elif char_at(current + 1) in ("E", "I", "Y"):
                    primary.append("J")
                    secondary.append("K")
                    current += 2
                else:
                    primary.append("K")
                    secondary.append("K")
                    current += 1
            elif ch == "H":
                if (current == 0 or char_at(current - 1) in ("A", "E", "I", "O", "U", "Y")) and char_at(current + 1) in ("A", "E", "I", "O", "U", "Y"):
                    primary.append("H")
                    secondary.append("H")
                    current += 2
                else:
                    current += 1
            elif ch == "J":
                if substr(current, 4) == "JOSE" or substr(current, 3) == "SAN":
                    primary.append("H")
                    secondary.append("H")
                else:
                    primary.append("J")
                    secondary.append("A")
                if char_at(current + 1) == "J":
                    current += 2
                else:
                    current += 1
            elif ch == "K":
                primary.append("K")
                secondary.append("K")
                if char_at(current + 1) == "K":
                    current += 2
                else:
                    current += 1
            elif ch == "L":
                primary.append("L")
                secondary.append("L")
                if char_at(current + 1) == "L":
                    current += 2
                else:
                    current += 1
            elif ch == "M":
                primary.append("M")
                secondary.append("M")
                if char_at(current + 1) == "M":
                    current += 2
                else:
                    current += 1
            elif ch == "N":
                primary.append("N")
                secondary.append("N")
                if char_at(current + 1) == "N":
                    current += 2
                else:
                    current += 1
            elif ch == "P":
                if char_at(current + 1) == "H":
                    primary.append("F")
                    secondary.append("F")
                    current += 2
                elif char_at(current + 1) in ("P", "B"):
                    primary.append("P")
                    secondary.append("P")
                    current += 2
                else:
                    primary.append("P")
                    secondary.append("P")
                    current += 1
            elif ch == "Q":
                primary.append("K")
                secondary.append("K")
                if char_at(current + 1) == "Q":
                    current += 2
                else:
                    current += 1
            elif ch == "R":
                primary.append("R")
                secondary.append("R")
                if char_at(current + 1) == "R":
                    current += 2
                else:
                    current += 1
            elif ch == "S":
                if substr(current, 2) == "SH":
                    primary.append("X")
                    secondary.append("X")
                    current += 2
                elif substr(current, 3) in ("SIO", "SIA"):
                    primary.append("S")
                    secondary.append("X")
                    current += 3
                elif substr(current, 2) == "SC":
                    if char_at(current + 2) in ("E", "I", "Y"):
                        primary.append("S")
                        secondary.append("S")
                        current += 3
                    else:
                        primary.append("SK")
                        secondary.append("SK")
                        current += 3
                else:
                    primary.append("S")
                    secondary.append("S")
                    if char_at(current + 1) in ("S", "Z"):
                        current += 2
                    else:
                        current += 1
            elif ch == "T":
                if substr(current, 4) == "TION":
                    primary.append("X")
                    secondary.append("X")
                    current += 4
                elif substr(current, 3) in ("TIA", "TCH"):
                    primary.append("X")
                    secondary.append("X")
                    current += 3
                elif substr(current, 2) == "TH":
                    primary.append("0")  # theta
                    secondary.append("T")
                    current += 2
                else:
                    primary.append("T")
                    secondary.append("T")
                    if char_at(current + 1) in ("T", "D"):
                        current += 2
                    else:
                        current += 1
            elif ch == "V":
                primary.append("F")
                secondary.append("F")
                if char_at(current + 1) == "V":
                    current += 2
                else:
                    current += 1
            elif ch == "W":
                if substr(current, 2) == "WR":
                    primary.append("R")
                    secondary.append("R")
                    current += 2
                elif current == 0 and char_at(current + 1) in ("A", "E", "I", "O", "U", "Y"):
                    primary.append("A")
                    secondary.append("F")
                    current += 1
                else:
                    current += 1
            elif ch == "X":
                if not (current == length - 1 and substr(current - 2, 2) in ("IA", "AU")):
                    primary.append("KS")
                    secondary.append("KS")
                if char_at(current + 1) in ("C", "X"):
                    current += 2
                else:
                    current += 1
            elif ch == "Z":
                if char_at(current + 1) == "H":
                    primary.append("J")
                    secondary.append("J")
                    current += 2
                else:
                    primary.append("S")
                    secondary.append("S")
                if char_at(current + 1) == "Z":
                    current += 2
                else:
                    current += 1
            else:
                current += 1

        p_str = "".join(primary)[:6]
        s_str = "".join(secondary)[:6]
        return (p_str, s_str)
