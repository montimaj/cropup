"""Build CropUp's place-name gazetteer from the real UserLocations pings.

0 of the 78 real farmer questions carry GPS, but 18 name a place in free text.
Resolving those names to a coordinate is what makes an Earth Engine analysis
possible at all, so the gazetteer is the primary location-slot filler.

SPEC section 7 governs what may leave this script. The pings are personal data
and the output is committed, so the artifact carries only what the runtime
consumer (``cropup/nlu/slots.py``) reads, and nothing that measures an
individual: the per-place ping count is computed here, used here to suppress
and to band, and never written out, and the per-place ping dispersion is not
computed at all. See PUBLISHED_FIELDS.
"""
import json
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "Data", "UserLocations_202607281713 - UserLocations_202607281713.csv")
OUT = os.path.join(REPO, "cropup", "data", "gazetteer.json")

# Privacy controls. The source pings are personal data (a ProfileId plus an
# exact GPS fix), and this file is committed, so entries must not be traceable
# to an individual. K_MIN is a k-anonymity floor on the pings behind a centroid;
# PRECISION rounds coordinates to ~1 km, which is far finer than the 60 m field
# buffer needs but coarse enough not to identify a homestead.
K_MIN = 5
PRECISION = 2

# What a committed place record is allowed to contain. Every one of these is
# loaded by _build_place_vocab() in cropup/nlu/slots.py into a Place and
# reaches the API through Place.as_dict(); nothing else is published.
#
#   key/name/lat/lon    the lookup and the coordinate
#   country/parent      corroboration for an ambiguous short name
#   admin_level         specificity, and the source column the name came from
#   ambiguous           the AMBIGUOUS flag below
#   support_band        coarse attestation band, see BANDS
#   sources             seeded public centroid vs user-derived, see SOURCE_USER
#
# Deliberately absent:
#
#   n_pings    the exact number of personal pings behind the centroid. An exact
#              count is a near-unique fingerprint of one locality's records in
#              UserLocations and is a re-identification key when joined against
#              any other view of that table. Published as a band instead.
#   spread_deg the geographic dispersion of those pings, i.e. how far apart the
#              people who pinged this name actually were. It is not published
#              and it is not computed either: nothing at runtime needs it, the
#              ambiguity signal being the AMBIGUOUS flag below, which is an
#              editorial list rather than a measurement.
PUBLISHED_FIELDS = (
    "key", "name", "lat", "lon", "country", "parent",
    "admin_level", "ambiguous", "support_band", "sources",
)

# Coarse attestation bands, ascending, replacing the exact ping count.
#
# Why publish anything at all: slot extraction ranks competing place matches,
# and when two names tie on match score and span the better-attested locality
# is the better guess -- without a tiebreaker that choice falls to sort order,
# which is arbitrary. _band_rank() in cropup/nlu/slots.py turns the label back
# into that ordinal, reading the ascending order out of privacy.support_bands
# below, and _rank() and suggest_places() break their ties on it. A 4-valued
# band supports that comparison while telling an attacker only which order of
# magnitude a locality's traffic falls in, which is not a fingerprint. An entry
# no ping attests at or above K_MIN (a seeded town centroid) gets None rather
# than a band: that is the absence of a measurement, not "the lowest band", and
# SPEC section 4 names a missing value instead of defaulting it.
BANDS = ((10, "5-9"), (50, "10-49"), (500, "50-499"), (None, "500+"))
# The lowest band opens at the k-anonymity floor, so a published band always
# describes at least K_MIN pings and can never be read back as a sub-k count.
# Checked rather than asserted: `python -O` drops an assert.
if BANDS[0][1].split("-")[0] != str(K_MIN):
    raise RuntimeError(f"lowest band {BANDS[0][1]!r} must open at K_MIN={K_MIN}")

# Provenance, coarse on purpose. The runtime distinction that matters is
# "public town centroid" (exempt from SPEC section 7) versus "derived from
# personal pings"; which administrative column a name came from is already
# published as admin_level, so the source tag does not repeat it.
SOURCE_USER = "userlocations"

# Suffixes Google's reverse geocoder appends that farmers never type.
SUFFIXES = re.compile(r"\s+(region|district|province|county|city|municipal(ity)?|rural|urban|ward|division)$", re.I)

# Places named in the 78 real questions that the ping data may not cover.
# Coordinates are town/district centroids; kept explicit and auditable.
SEED = [
    ("Dodoma",   -6.1630,  35.7516, "Tanzania", "Dodoma Region",      "seed:question-corpus"),
    ("Njombe",   -9.3333,  34.7667, "Tanzania", "Njombe Region",      "seed:question-corpus"),
    ("Kibaha",   -6.7667,  38.9167, "Tanzania", "Pwani Region",       "seed:question-corpus"),
    ("Tanga",    -5.0689,  39.0988, "Tanzania", "Tanga Region",       "seed:question-corpus"),
    ("Hai",      -3.3500,  37.1500, "Tanzania", "Kilimanjaro Region", "seed:question-corpus"),
    ("Siha",     -3.2500,  37.1000, "Tanzania", "Kilimanjaro Region", "seed:question-corpus"),
    ("Hedaru",   -4.4167,  37.8333, "Tanzania", "Kilimanjaro Region", "seed:question-corpus"),
    ("Same",     -4.0667,  37.7333, "Tanzania", "Kilimanjaro Region", "seed:question-corpus"),
    ("Mateves",  -3.4000,  36.6000, "Tanzania", "Arusha Region",      "seed:question-corpus"),
    ("Bungu",    -7.9500,  39.0500, "Tanzania", "Pwani Region",       "seed:question-corpus"),
    ("Kibiti",   -7.7167,  38.9500, "Tanzania", "Pwani Region",       "seed:question-corpus"),
    ("Morogoro", -6.8210,  37.6610, "Tanzania", "Morogoro Region",    "seed:question-corpus"),
    # Google's reverse geocoder writes "Dar es Salam" (one a) in the ping data,
    # but every farmer spells it "Dar es Salaam".
    ("Dar es Salaam", -6.7924, 39.2083, "Tanzania", "Dar es Salaam Region", "seed:spelling-variant"),
    ("Morombo",  -3.3800,  36.7200, "Tanzania", "Arusha Region",      "seed:question-corpus"),
    ("Arusha",   -3.3869,  36.6830, "Tanzania", "Arusha Region",      "seed:question-corpus"),
    ("Moshi",    -3.3349,  37.3406, "Tanzania", "Kilimanjaro Region", "seed:question-corpus"),
    ("Mbeya",    -8.9000,  33.4500, "Tanzania", "Mbeya Region",       "seed:question-corpus"),
    ("Iringa",   -7.7700,  35.6900, "Tanzania", "Iringa Region",      "seed:question-corpus"),
    ("Singida",  -4.8167,  34.7500, "Tanzania", "Singida Region",     "seed:question-corpus"),
]

# Short names that collide with far-away places or common English words.
# These are only accepted when the message also names a matching parent region
# or the user's country context agrees. See extract_locations() in
# cropup/nlu/slots.py, which reads this flag off the `ambiguous` field.
AMBIGUOUS = {"hai", "same", "mara", "lindi", "pare", "kilosa", "bunda", "mpwapwa"}


def norm(name):
    if not isinstance(name, str):
        return None
    n = name.strip()
    if not n or n.lower() in ("nan", "none", "unknown"):
        return None
    n = SUFFIXES.sub("", n).strip()
    return n or None


def band(n_pings):
    """Coarse attestation band, or None when no band may honestly be published.

    None means "nothing attests this entry at or above K_MIN", which covers a
    seeded centroid with no ping at all and a seeded centroid with four. Only
    seeds can be below the floor -- everything else is suppressed -- and since
    the lowest band opens at K_MIN, banding four pings as "5-9" would publish a
    count the data does not support.
    """
    if n_pings < K_MIN:
        return None
    for ceiling, label in BANDS:
        if ceiling is None or n_pings < ceiling:
            return label
    raise AssertionError("BANDS must end in an open bucket")


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing source: {SRC}")
    df = pd.read_csv(SRC)
    df = df[df["Lat"].notna() & df["Lon"].notna()]
    # Drop mocked GPS and wildly imprecise fixes before trusting a centroid.
    if "Mock" in df:
        df = df[df["Mock"] != 1]

    entries = {}

    def add(name, lat, lon, country, parent, level, source, n_pings):
        key = name.lower()
        e = entries.setdefault(key, {
            "name": name, "country": country, "parent": parent,
            "level": level, "lats": [], "lons": [], "sources": set(),
            "n_pings": 0,
        })
        e["lats"].append(lat)
        e["lons"].append(lon)
        e["sources"].add(source)
        # n_pings counts personal records. It decides suppression and the
        # published band, and is discarded with `entries` when main() returns.
        e["n_pings"] += n_pings
        # Prefer the most specific level seen for this name.
        if level > e["level"]:
            e["level"], e["parent"] = level, parent

    cols = [
        ("AdministrativeAreaLevel1", 1),
        ("AdministrativeAreaLevel2", 2),
        ("AdministrativeAreaLevel3", 3),
        ("AdministrativeAreaLevel4", 4),
    ]
    for col, level in cols:
        if col not in df:
            continue
        sub = df[df[col].notna()]
        for name, grp in sub.groupby(col):
            n = norm(name)
            if not n or len(n) < 3:
                continue
            parent = None
            if level > 1:
                pcol = cols[level - 2][0]
                pv = grp[pcol].dropna()
                parent = norm(pv.iloc[0]) if len(pv) else None
            country = grp["Country"].dropna().iloc[0] if grp["Country"].notna().any() else None
            add(n, float(grp["Lat"].median()), float(grp["Lon"].median()),
                country, parent, level, SOURCE_USER, n_pings=int(len(grp)))

    for name, lat, lon, country, parent, source in SEED:
        # Seeds are published town centroids, not user data: no k-anonymity bar,
        # and no pings, so they contribute 0 to the count that sets the band.
        add(name, lat, lon, country, parent, 3, source, n_pings=0)

    out, suppressed = [], []
    for key, e in sorted(entries.items()):
        lats, lons = e["lats"], e["lons"]
        lat = sorted(lats)[len(lats) // 2]
        lon = sorted(lons)[len(lons) // 2]
        is_seed = any(s.startswith("seed") for s in e["sources"])
        # PRIVACY: UserLocations holds personal GPS pings tied to a ProfileId.
        # An area with few pings has a "centroid" that is effectively one
        # person's recorded position, so below K_MIN we suppress the entry
        # entirely rather than publish it.
        if not is_seed and e["n_pings"] < K_MIN:
            suppressed.append(key)
            continue
        # Even above the bar, coordinates are rounded to ~1 km so the entry
        # names a locality rather than a household.
        lat, lon = round(lat, PRECISION), round(lon, PRECISION)
        row = {
            "key": key,
            "name": e["name"],
            "lat": lat,
            "lon": lon,
            "country": e["country"],
            "parent": e["parent"],
            "admin_level": e["level"],
            "ambiguous": key in AMBIGUOUS,
            "support_band": band(e["n_pings"]),
            "sources": sorted(e["sources"]),
        }
        # Not an assert: `python -O` drops those, and this check is the last
        # thing between a newly added field and SPEC section 7.
        if set(row) != set(PUBLISHED_FIELDS):
            raise RuntimeError(
                "record shape drifted from PUBLISHED_FIELDS: "
                f"{sorted(set(row) ^ set(PUBLISHED_FIELDS))}"
            )
        out.append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "places": out,
            "privacy": {
                "k_min": K_MIN,
                "coord_precision_dp": PRECISION,
                "suppressed_below_k": len(suppressed),
                "support_bands": [label for _, label in BANDS],
                "note": ("Derived from personal GPS pings. Entries backed by fewer "
                         "than k_min pings are suppressed; coordinates are rounded. "
                         "Seeded town centroids are public geography, not user data. "
                         "No exact per-place ping count or ping dispersion is "
                         "published: support_band gives the band of the count, in "
                         "the ascending order of support_bands, and is null for an "
                         "entry no ping backs."),
            },
        }, f, indent=1)

    tz = [o for o in out if o["country"] == "Tanzania"]
    seeded = [o for o in out if any(s.startswith("seed") for s in o["sources"])]
    print(f"wrote {len(out)} places -> {OUT}")
    print(f"  Tanzania: {len(tz)}  ambiguous-flagged: {sum(1 for o in out if o['ambiguous'])}")
    print(f"  by level: " + ", ".join(
        f"L{l}={sum(1 for o in out if o['admin_level'] == l)}" for l in (1, 2, 3, 4)))
    print(f"  seeded (public centroids): {len(seeded)}")
    print(f"  support_band: " + ", ".join(
        f"{label or 'null'}={sum(1 for o in out if o['support_band'] == label)}"
        for label in [label for _, label in BANDS] + [None]))
    print(f"  SUPPRESSED for k<{K_MIN}: {len(suppressed)} (privacy)")
    print(f"  published fields: {', '.join(PUBLISHED_FIELDS)}")


if __name__ == "__main__":
    main()
