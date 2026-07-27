"""checktags.py — 20-second ground-truth check: what does OSM actually say
about the three facilities that keep coming back for Psychiatry?

Run:  python checktags.py
"""
import requests

QUERY = '''
[out:json][timeout:25];
(
  nwr["name"~"Klinik Guhen|Rainbow Child|Neuro and Parkinson",i]
     (around:12000,12.93212,77.70449);
);
out center tags;
'''

r = requests.post("https://overpass-api.de/api/interpreter",
                  data={"data": QUERY},
                  headers={"User-Agent": "CareRoute-debug/1.0"}, timeout=30)
r.raise_for_status()
for el in r.json().get("elements", []):
    t = el.get("tags", {})
    print(f"{el['type']:5} {t.get('name', '?')}")
    print(
        f"      healthcare:speciality = {t.get('healthcare:speciality', '(no tag)')}")
    others = {k: v for k, v in t.items()
              if k not in ("name", "healthcare:speciality")}
    print(f"      other tags: {others}\n")
