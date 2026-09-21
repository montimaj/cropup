"""Build CropUp's crop vocabulary: canonical names, aliases and backend coverage.

Three downstream consumers disagree about crop names, so the vocabulary records
which of them actually knows each crop:
  - MaxEnt / CropSuite suitability  (98 crops)  -> crop_selection
  - the disease library rules engine (102 crops) -> plant_health / diagnosis
  - free text from farmers, including Swahili   -> every intent
A crop the farmer names but no backend covers must degrade honestly rather than
silently fall through to the "Generic" rule set without saying so.
"""
import json
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "Data", "googlebuildathonfarmers-main")
OUT = os.path.join(REPO, "cropup", "data", "crops.json")

# Swahili and East African common names seen in the real question corpus and in
# ordinary Tanzanian usage. Mapped to the canonical English the backends expect.
SWAHILI = {
    "Maize": ["mahindi", "mhindi", "corn"],
    "Rice": ["mpunga", "mchele", "wali"],
    "Beans": ["maharage", "maharagwe", "common bean"],
    "Cassava": ["muhogo", "mihogo", "manioc", "tapioca"],
    "Potato": ["viazi", "viazi mviringo", "irish potato"],
    "Sweet Potato": ["viazi vitamu", "kiazi kitamu"],
    "Tomato": ["nyanya"],
    "Wild Tomato": ["nyanya chungu", "african eggplant", "bitter tomato"],
    "Onion": ["vitunguu", "kitunguu"],
    "Garlic": ["kitunguu saumu", "kitunguu swaumu"],
    "Sorghum": ["mtama"],
    "Finger Millet": ["ulezi", "wimbi"],
    "Pearl Millet": ["uwele"],
    "Millet": ["mtama mdogo"],
    "Groundnut": ["karanga", "njugu", "peanut", "peanuts"],
    "Sunflower": ["alizeti"],
    "Cotton": ["pamba"],
    "Coffee": ["kahawa"],
    "Tea": ["chai"],
    "Banana": ["ndizi", "mgomba"],
    "Plantain": ["ndizi za kupika"],
    "Mango": ["embe", "miembe"],
    "Avocado": ["parachichi", "maparachichi"],
    "Orange": ["chungwa", "machungwa"],
    "Lemon": ["limao", "ndimu"],
    "Coconut": ["nazi", "minazi"],
    "Sisal": ["mkonge"],
    "Sesame": ["ufuta", "simsim", "sim sim"],
    "Pigeon Pea": ["mbaazi"],
    "Cowpea": ["kunde"],
    "Mung Bean": ["choroko"],
    "Lentil": ["dengu"],
    "Pea": ["njegere"],
    "Chickpea": ["dengu za kihindi"],
    "Spinach": ["mchicha"],
    "Cabbage": ["kabichi"],
    "Chinese Cabbage": ["chinese", "chainizi", "pak choi", "bok choy"],
    "Carrot": ["karoti"],
    "Cucumber": ["tango", "matango"],
    "Watermelon": ["tikiti maji", "tikitimaji"],
    "Pumpkin": ["maboga", "boga"],
    "Okra": ["bamia"],
    "Eggplant": ["biringanya", "brinjal", "aubergine"],
    "Pepper": ["pilipili hoho", "bell pepper", "capsicum", "hoho"],
    "Sugarcane": ["miwa"],
    "Papaya": ["papai", "mapapai", "pawpaw"],
    "Pineapple": ["nanasi", "mananasi"],
    "Passion Fruit": ["passion", "pasheni"],
    "Guava": ["mapera", "pera"],
    "Cashew": ["korosho", "mkorosho", "cashew nut"],
    "Sugar Beet": ["beetroot"],
    "Soybean": ["soya", "soy"],
    "Wheat": ["ngano"],
    "Barley": ["shayiri"],
    "Yam": ["viazi vikuu"],
    "Ginger": ["tangawizi"],
    "Turmeric": ["manjano", "binzari"],
    "Black Pepper": ["pilipili manga"],
    "Moringa": ["mlonge", "mrongo", "drumstick tree"],
    "Amaranth": ["mchicha wa kienyeji"],
}


def norm_key(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    # MaxEnt crop list (also the CropSuite-facing vocabulary).
    maxent_src = os.path.join(REF, "app", "data_sources", "maxent.py")
    m = re.search(r"MAXENT_CROP_NAMES = \{(.*?)\n\}", open(maxent_src).read(), re.S)
    maxent = {common for _sci, common in re.findall(r'"([a-z_]+)":\s*"([^"]+)"', m.group(1))}

    # Disease-library crop list (drives the rules engine).
    dl = pd.read_csv(os.path.join(REF, "data", "disease_library.csv"))
    disease = {c for c in dl["crop"].dropna().unique() if c != "Generic"}

    canonical = sorted(maxent | disease | set(SWAHILI))

    crops = []
    for name in canonical:
        aliases = set(SWAHILI.get(name, []))
        aliases.add(name.lower())
        # Cheap morphological variants farmers actually type.
        low = name.lower()
        if not low.endswith("s"):
            aliases.add(low + "s")
        if low.endswith("y"):
            aliases.add(low[:-1] + "ies")
        crops.append({
            "name": name,
            "key": norm_key(name),
            "aliases": sorted(a for a in aliases if a),
            "in_suitability": name in maxent,
            "in_disease_library": name in disease,
            "rules_fallback": "Generic" if name not in disease else None,
        })

    both = [c for c in crops if c["in_suitability"] and c["in_disease_library"]]
    only_s = [c for c in crops if c["in_suitability"] and not c["in_disease_library"]]
    only_d = [c for c in crops if c["in_disease_library"] and not c["in_suitability"]]
    neither = [c for c in crops if not c["in_suitability"] and not c["in_disease_library"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "crops": crops,
            "coverage": {
                "total": len(crops),
                "in_both": len(both),
                "suitability_only": len(only_s),
                "disease_library_only": len(only_d),
                "neither": len(neither),
            },
        }, f, indent=1)

    n_alias = sum(len(c["aliases"]) for c in crops)
    print(f"wrote {len(crops)} crops, {n_alias} aliases -> {OUT}")
    print(f"  in both backends:      {len(both)}")
    print(f"  suitability only:      {len(only_s)}")
    print(f"  disease library only:  {len(only_d)}")
    print(f"  NEITHER (degrade):     {len(neither)} -> {[c['name'] for c in neither]}")


if __name__ == "__main__":
    main()
