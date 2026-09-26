"""Cleaning rules for business names and addresses.

Everything is rule based and country agnostic: the same rules run on US, India,
France or any other country label. The only country knowledge is a table of
state/region names and their common codes (incl. native-script spellings), used
to put the state in its own field.
"""
import re

from unidecode import unidecode

# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
MOJIBAKE_RE = re.compile(r"[ÂâÃ][\u0080-\u009f]+|[\u0080-\u009f]")
ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}")          # l.l.c. -> llc
APOS_RE = re.compile(r"['`’]")
NONALNUM_RE = re.compile(r"[^a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
VOWEL_RE = re.compile(r"[aeiouyh]")
REPEAT_RE = re.compile(r"(.)\1+")
LEET = str.maketrans("013458", "oleasb")


def to_ascii(s):
    if not s:
        return ""
    s = MOJIBAKE_RE.sub(" ", s)
    if not s.isascii():
        s = unidecode(s)
    return s.lower()


def alnum_tokens(s):
    return NONALNUM_RE.sub(" ", s).split()


def skel(tok):
    """Consonant skeleton: survives transliteration and vowel typos.
    raam -> rm, ram -> rm, maarkettiNg -> mrktng, marketing -> mrktng."""
    if not tok or tok[0].isdigit():
        return tok
    t = tok.replace("ph", "f").replace("ck", "k").replace("q", "k").replace("c", "k")
    t = t.replace("z", "s").replace("w", "v").replace("x", "ks")
    t = t[0] + VOWEL_RE.sub("", t[1:])
    return REPEAT_RE.sub(r"\1", t)


def fix_leet(tok):
    """c0mpany -> company, 8usiness -> business (one digit inside a word)."""
    if len(tok) >= 4:
        nd = sum(c.isdigit() for c in tok)
        if nd == 1 and len(tok) - nd >= 3:
            return tok.translate(LEET)
    return tok


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------
NAME_CANON = {
    "limited": "ltd", "private": "pvt", "corporation": "corp", "incorporated": "inc",
    "company": "co", "compagnie": "co", "cie": "co", "international": "intl",
    "brothers": "bros", "centre": "center", "services": "service", "svcs": "service",
    "svc": "service", "technologies": "tech", "technology": "tech",
    "enterprise": "enterprises", "associates": "assoc", "associate": "assoc",
    "manufacturing": "mfg", "management": "mgmt", "grp": "group",
    "holding": "holdings", "hldgs": "holdings", "societe": "ste", "saint": "st",
    "etablissements": "ets", "establishments": "ets", "solution": "solutions",
    "consultant": "consultants", "industry": "industries", "venture": "ventures",
    "partner": "partners", "system": "systems", "product": "products",
}
LEGAL = {
    "ltd", "pvt", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pllc", "pc",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "gmbh", "opc", "pty",
    "ag", "nv", "bv",
}
# legal words written in Indian scripts, recognised through their skeleton
LEGAL_SKEL = {"prvt": "pvt", "lmtd": "ltd", "krprtn": "corp", "kmpn": "co",
              "inkrprtd": "inc", "elp": "llp"}
NAME_STOP = {"the", "and", "of", "a"}
GENERIC = {
    "service", "group", "partners", "center", "holdings", "solutions", "enterprises",
    "ventures", "assoc", "intl", "global", "industries", "trading", "consultants",
    "consulting", "india", "france", "usa", "systems", "mgmt", "tech", "products",
}
DBA_RE = re.compile(
    r"\b(?:doing business as|trading as|d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|aka|formerly)\b")
MS_RE = re.compile(r"\bm\s*/\s*s\b\.?")
DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|in|net|org|fr|biz|info|io|co|us)$")


def _name_tokens(s):
    s = ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    s = APOS_RE.sub("", s).replace("&", " and ").replace("+", " and ")
    out = []
    for t in alnum_tokens(s):
        t = fix_leet(t)
        t = NAME_CANON.get(t, t)
        if t not in LEGAL and len(t) >= 5:
            t = LEGAL_SKEL.get(skel(t), t)
        elif t not in LEGAL and t == "elelpii":
            t = "llp"
        out.append(t)
    return out


def _core(tokens):
    core = [t for t in tokens if t not in LEGAL and t not in NAME_STOP]
    return core or [t for t in tokens if t not in NAME_STOP] or tokens


def clean_name(raw):
    s = to_ascii(raw).strip()
    s = MS_RE.sub(" ", s)
    is_domain = 0
    m = DOMAIN_RE.match(SPACE_RE.sub("", s))
    if m:
        s, is_domain = m.group(1).replace("-", " "), 1

    parts = DBA_RE.split(s, maxsplit=1)
    main = _name_tokens(parts[0])
    alt = _name_tokens(parts[1]) if len(parts) > 1 else []
    if not main and alt:
        main, alt = alt, []

    core = _core(main)
    legal = sorted({t for t in main if t in LEGAL})
    key = [t for t in core if t not in GENERIC] or core
    alt_core = _core(alt) if alt else []
    return {
        "name_clean": " ".join(main),
        "name_core": " ".join(core),
        "name_key": " ".join(key),
        "name_alt": " ".join(alt_core),
        "name_skel": " ".join(skel(t) for t in core),
        "name_nospace": "".join(core),
        "name_legal": " ".join(legal),
        "is_domain": is_domain,
    }


# --------------------------------------------------------------------------
# addresses
# --------------------------------------------------------------------------
_STATES = {
    # US
    "al": ["alabama"], "ak": ["alaska"], "az": ["arizona"], "ar": ["arkansas"],
    "ca": ["california"], "co": ["colorado"], "ct": ["connecticut"], "de": ["delaware"],
    "dc": ["district of columbia", "washington dc"], "fl": ["florida"], "ga": ["georgia", "goa"],
    "hi": ["hawaii"], "id": ["idaho"], "il": ["illinois"], "in": ["indiana"], "ia": ["iowa"],
    "ks": ["kansas"], "ky": ["kentucky"], "la": ["louisiana"], "me": ["maine"],
    "md": ["maryland"], "ma": ["massachusetts"], "mi": ["michigan"], "mn": ["minnesota", "manipur"],
    "ms": ["mississippi"], "mo": ["missouri"], "mt": ["montana"], "ne": ["nebraska"],
    "nv": ["nevada"], "nh": ["new hampshire"], "nj": ["new jersey"], "nm": ["new mexico"],
    "ny": ["new york"], "nc": ["north carolina"], "nd": ["north dakota"], "oh": ["ohio"],
    "ok": ["oklahoma"], "or": ["oregon", "odisha", "orissa", "od", "ଓଡ଼ିଶା"],
    "pa": ["pennsylvania"], "ri": ["rhode island"], "sc": ["south carolina"],
    "sd": ["south dakota"], "tn": ["tennessee", "tamil nadu", "தமிழ்நாடு"],
    "tx": ["texas"], "ut": ["utah"], "vt": ["vermont"], "va": ["virginia"],
    "wa": ["washington"], "wv": ["west virginia"], "wi": ["wisconsin"], "wy": ["wyoming"],
    "pr": ["puerto rico"],
    # India
    "mh": ["maharashtra", "महाराष्ट्र"],
    "dl": ["delhi", "nct of delhi", "दिल्ली"],
    "up": ["uttar pradesh", "उत्तर प्रदेश"],
    "ka": ["karnataka", "ಕರ್ನಾಟಕ"],
    "gj": ["gujarat", "ગુજરાત"],
    "wb": ["west bengal", "পশ্চিমবঙ্গ"],
    "tg": ["telangana", "ts", "తెలంగాణ"],
    "hr": ["haryana", "हरियाणा"],
    "rj": ["rajasthan", "राजस्थान"],
    "kl": ["kerala", "കേരളം"],
    "br": ["bihar", "बिहार"],
    "mp": ["madhya pradesh", "मध्य प्रदेश"],
    "ap": ["andhra pradesh", "ఆంధ్రప్రదేశ్"],
    "pb": ["punjab", "ਪੰਜਾਬ"],
    "jh": ["jharkhand"], "cg": ["chhattisgarh", "chattisgarh", "ct"],
    "uk": ["uttarakhand", "uttaranchal"], "hp": ["himachal pradesh"],
    "jk": ["jammu and kashmir", "jammu kashmir"], "as": ["assam"], "ch": ["chandigarh"],
    "py": ["puducherry", "pondicherry"], "tr": ["tripura"], "ml": ["meghalaya"],
    "nl": ["nagaland"], "arp": ["arunachal pradesh"], "mz": ["mizoram"], "sk": ["sikkim"],
    # France (regions; departments mapped to their region)
    "hdf": ["hauts de france", "nord", "pas de calais", "somme", "aisne", "oise"],
    "naq": ["nouvelle aquitaine", "gironde", "landes", "pyrenees atlantiques", "charente maritime"],
    "pdl": ["pays de la loire", "loire atlantique", "vendee", "maine et loire", "sarthe", "mayenne"],
    "idf": ["ile de france", "paris"], "ara": ["auvergne rhone alpes"], "occ": ["occitanie"],
    "ges": ["grand est"], "pac": ["provence alpes cote d azur", "paca"], "bre": ["bretagne"],
    "nor": ["normandie"], "cvl": ["centre val de loire"], "bfc": ["bourgogne franche comte"],
    "cor": ["corse"],
}
STATE_MAP = {}
for _code, _names in _STATES.items():
    STATE_MAP[_code] = _code
    for _n in _names:
        STATE_MAP[" ".join(alnum_tokens(to_ascii(_n)))] = _code
COUNTRY_WORDS = {"india", "usa", "us", "united states", "united states of america", "france"}

ADDR_CANON = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "bvd": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "highway": "hwy", "parkway": "pkwy", "place": "pl", "square": "sq",
    "circle": "cir", "trail": "trl", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "unit", "apt": "unit", "flat": "unit", "suite": "unit", "ste": "unit",
    "room": "unit", "floor": "fl", "flr": "fl", "building": "bldg", "mount": "mt",
    "fort": "ft", "sector": "sec", "extension": "ext", "extn": "ext", "colony": "col",
    "ngr": "nagar", "market": "mkt", "route": "rte", "allee": "all", "chemin": "chem",
    "impasse": "imp", "faubourg": "fbg", "fg": "fbg", "r": "rue", "bengaluru": "bangalore",
    "gurugram": "gurgaon", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
}
ADDR_STOP = {"of", "the", "de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "a"}
LANDMARK_RE = re.compile(
    r"\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|close to|landmark|land mark)\b.*$")
LABEL_RE = re.compile(
    r"\b(?:h\s*\.?\s*no|(?:house|door|plot|flat|shop|office|bldg|building|survey|sy|kh|khasra|gali|room|site|ward|property)\s*\.?\s*no|nos|no|num|number)\b\.?")
NUM_JOIN_RE = re.compile(r"(?<=\d)\s*[-/]\s*(?=[a-z0-9])|(?<=[a-z0-9])\s*[-/]\s*(?=\d)")
PIN_SPLIT_RE = re.compile(r"(?<!\d)(\d{3}) (\d{3})(?!\d)")
CDP_RE = re.compile(r"\s*cdp\b")
DIGITS_RE = re.compile(r"\d+")
STANDALONE_NUM_RE = re.compile(r"(?<![\w\-/])\d+(?![\w\-/])")


def clean_address(raw):
    s = to_ascii(raw).strip()
    if not s:
        return {"addr_clean": "", "house_no": "", "postcode": "", "state": "",
                "numbers": "", "landmark": "", "addr_missing": 1}
    s = PIN_SPLIT_RE.sub(r"\1\2", s)
    s = CDP_RE.sub(" ", s)

    state, landmarks, tokens, numbers = "", [], [], []
    postcode = ""
    for part in s.split(","):
        norm = " ".join(alnum_tokens(part))
        if not norm or norm in COUNTRY_WORDS:
            continue
        if norm in STATE_MAP:
            state = state or STATE_MAP[norm]
            continue
        m = LANDMARK_RE.search(norm)
        if m:
            landmarks.append(m.group(0))
            part = part[: max(0, len(part) - len(m.group(0)))] if m.start() else ""
            norm = norm[: m.start()]
            if not norm.strip():
                continue
        part = LABEL_RE.sub(" ", part)
        for d in STANDALONE_NUM_RE.findall(part):
            if 5 <= len(d) <= 6 and not d.startswith("00"):
                postcode = d
        numbers.extend(str(int(d)) for d in DIGITS_RE.findall(part))
        part = NUM_JOIN_RE.sub("", part)
        for t in alnum_tokens(APOS_RE.sub("", part)):
            if t.isdigit():
                t = str(int(t))
            else:
                t = ADDR_CANON.get(t, t)
            if t not in ADDR_STOP:
                tokens.append(t)

    first_num = next((t for t in tokens if any(c.isdigit() for c in t)), "")
    if len(postcode) == 5 and first_num == str(int(postcode)):
        postcode = ""          # a leading 5-digit number is a house number, not a ZIP
    pc = str(int(postcode)) if postcode else ""
    house = next((t for t in tokens if any(c.isdigit() for c in t) and t != pc), "")
    return {
        "addr_clean": " ".join(tokens),
        "house_no": house,
        "postcode": postcode,
        "state": state,
        "numbers": " ".join(dict.fromkeys(numbers)),
        "landmark": " ".join(landmarks),
        "addr_missing": 0,
    }


# --------------------------------------------------------------------------
# blocking tokens
# --------------------------------------------------------------------------
def block_tokens(n, a):
    """Space separated, prefixed tokens used to find candidates."""
    toks = set()
    words = n["name_core"].split() + n["name_alt"].split()
    for t in words:
        if len(t) >= 2:
            toks.add("n_" + t)
            if not t.isdigit():
                k = skel(t)
                if len(k) >= 2:
                    toks.add("k_" + k)
    if len(n["name_nospace"]) >= 4:
        toks.add("z_" + n["name_nospace"])
    for t in a["addr_clean"].split():
        if len(t) >= 3 and not t.isdigit():
            toks.add("a_" + t)
    if a["house_no"]:
        toks.add("h_" + a["house_no"])
    if a["postcode"]:
        toks.add("p_" + a["postcode"])
    for d in a["numbers"].split():
        if len(d) >= 2:
            toks.add("d_" + d)
    return " ".join(sorted(toks))


def clean_record(name, addr):
    n = clean_name(name)
    a = clean_address(addr)
    out = {**n, **a}
    out["btok"] = block_tokens(n, a)
    return out


FIELDS = ["name_clean", "name_core", "name_key", "name_alt", "name_skel", "name_nospace",
          "name_legal", "is_domain", "addr_clean", "house_no", "postcode", "state",
          "numbers", "landmark", "addr_missing", "btok"]


if __name__ == "__main__":
    tests = [
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"),
        ("Quoavi Co doing business as Asset Building Committee", "315 80th Street, Chicago, IL"),
        ("ipower.com", "తెలంగాణ, Cbr Estates, Flat No 505 4Th Block, Hyderabad, 5-513/4"),
        ("Elite  + C0mpany", "B-59, DERAWAL NAGAR, Delhi"),
        ("Super 8usiness Pvt Ltd.", "MUMBAI Â\x80\x93 400 021, Near SBI ATM, MH"),
        ("M/s Agrotech Lubricants Limited Private", "Old No. Â\x80\x93 B -35 S/f Kh No. Â\x80\x93 829"),
        ("Fannie Trussell Seneca Inc.", "00709 Hackberry Saint, Tilden, Texas"),
        ("SAS Freres (Frànce) Amis", "Nº 50 R Pierre Larousse, Saint-nazaire, Loire-Atlantique"),
        ("Moyna's Coffee L.L.C.", "1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio"),
    ]
    for nm, ad in tests:
        r = clean_record(nm, ad)
        print(nm, "|", ad)
        for k in FIELDS:
            print(f"    {k:13s} {r[k]!r}")
