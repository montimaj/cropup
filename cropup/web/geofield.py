"""The exact field ``GET /api/geo/field`` promises (SPEC section 8).

The endpoint's job is to let the farmer see, before any quota is spent, *what
will be asked about*. Two honesty problems are in the way and both are answered
explicitly rather than smoothed over.

**1. The request is exact; the drawing is not.** What is actually sent to Earth
Engine is ``ee.Geometry.Point([lon, lat]).buffer(radius_m)`` -- three numbers.
Earth Engine turns that into a geodesic polygon on its own server, with a vertex
count it chooses. This process must not call Earth Engine to find out (SPEC
section 4.4 and the whole point of this endpoint is that it costs nothing), so
the ring returned here is computed locally, on a sphere, with a stated vertex
count, and is labelled ``approximate`` with the method that produced it. The
``request`` block carries the three numbers that are not approximate at all.

**2. There is not one radius.** ``field_radius_m`` (15 m by default) is the
farmer's field: it is what the vegetation and thermal legs reduce over. The
water leg reduces over 5 km because SMAP is an 11 km product and ESI is 5.5 km,
and the irrigation-regime leg over its own neighbourhood, because a point sample
of a coarse asset is a number about nothing. Showing only the 15 m disc would
tell the farmer their soil-moisture reading came from their field. So the
endpoint returns the field polygon *and* names every reduction footprint the
chosen analysis will use, each with the asset that forces it.

Nothing here imports ``ee``. The reduction radii are read from the ``geo``
modules' own constants, so they cannot drift from the code that uses them.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..config import Settings, get_settings
from ..dialog.slots import SlotBag
from ..geo import context as _context
from ..geo import suitability as _suitability
from ..geo import thermal as _thermal
from ..geo import vegetation as _vegetation
from ..geo import water as _water

__all__ = [
    "EARTH_RADIUS_M",
    "RING_VERTICES",
    "circle_ring",
    "effective_field_radius_m",
    "field_polygon",
    "reduction_footprints",
    "field_report",
]

#: WGS84 mean radius. The ring is a local drawing, not a measurement, and the
#: sphere is stated so a reader knows exactly what was assumed.
EARTH_RADIUS_M = 6_371_008.8

#: Vertices in the drawn ring. Declared, not tuned: it is reported alongside the
#: ring so the client knows the polygon's resolution instead of guessing.
RING_VERTICES = 64


def circle_ring(
    lat: float, lon: float, radius_m: float, *, vertices: int = RING_VERTICES
) -> list[list[float]]:
    """A closed ``[lon, lat]`` ring approximating a geodesic disc.

    Spherical destination-point formula, ``vertices`` evenly spaced bearings,
    first point repeated last so the ring closes as GeoJSON requires.
    """
    if radius_m <= 0:
        raise ValueError(f"radius must be positive metres, got {radius_m!r}")
    count = max(8, int(vertices))
    phi = math.radians(lat)
    lam = math.radians(lon)
    delta = float(radius_m) / EARTH_RADIUS_M
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_d, cos_d = math.sin(delta), math.cos(delta)

    ring: list[list[float]] = []
    for i in range(count):
        theta = 2.0 * math.pi * i / count
        sin_lat = sin_phi * cos_d + cos_phi * sin_d * math.cos(theta)
        lat2 = math.asin(max(-1.0, min(1.0, sin_lat)))
        lon2 = lam + math.atan2(
            math.sin(theta) * sin_d * cos_phi,
            cos_d - sin_phi * math.sin(lat2),
        )
        ring.append([round(math.degrees(lon2), 7), round(math.degrees(lat2), 7)])
    ring.append(list(ring[0]))
    return ring


def field_polygon(lat: float, lon: float, radius_m: float) -> dict[str, Any]:
    """A GeoJSON Feature for the field disc, honest about being a drawing."""
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [circle_ring(lat, lon, radius_m)]},
        "properties": {
            "role": "field",
            "radius_m": float(radius_m),
            "approximate": True,
            "vertices": RING_VERTICES,
            "method": (
                f"spherical destination-point ring, {RING_VERTICES} vertices, "
                f"R={EARTH_RADIUS_M:.1f} m; Earth Engine buffers the point itself "
                "and chooses its own vertex count, so this ring is for display"
            ),
        },
    }


def effective_field_radius_m(
    bag: SlotBag | None = None, settings: Settings | None = None
) -> tuple[float, str]:
    """The radius ``dispatch.authorise`` will actually put in the buffer, and why.

    ``dispatch.RunAuthorization`` is built with
    ``float(bag.field_radius_m or settings.field_radius_m)`` and that number is
    passed to the orchestrator as ``radius_m``. Reading ``settings`` alone -- as
    this module used to -- printed the environment's default next to a
    ``request`` block quoting the session's own radius, which is one endpoint
    disagreeing with itself about the one number it exists to report.
    """
    settings = settings or get_settings()
    chosen = getattr(bag, "field_radius_m", None) if bag is not None else None
    if isinstance(chosen, (int, float)) and not isinstance(chosen, bool) and chosen > 0:
        return float(chosen), "session"
    return float(settings.field_radius_m), "CROPUP_FIELD_RADIUS_M (no field size chosen yet)"


def reduction_footprints(
    analysis: str | None,
    settings: Settings | None = None,
    *,
    field_radius_m: float | None = None,
) -> list[dict[str, Any]]:
    """Every footprint the chosen analysis will reduce over, and why each is that size.

    A footprint larger than the field is not a mistake: it is the asset's own
    resolution asserting itself, and the farmer is entitled to know a soil
    moisture number came from 5 km around them rather than from their plot.

    ``field_radius_m`` is the radius this run will really use -- the session's,
    when it has one. Each row carries ``radius_source`` naming *what decides*
    that leg's footprint, because the legs do not agree and printing one number
    for all of them would be wrong for most of them: vegetation and thermal take
    the field radius, water ignores it entirely, suitability uses the
    neighbourhood setting, and soil and climate are read at the pixel.
    """
    settings = settings or get_settings()
    if field_radius_m is not None and field_radius_m > 0:
        field_r = float(field_radius_m)
        field_source = "the field radius this session will send"
    else:
        field_r = float(settings.field_radius_m)
        field_source = "CROPUP_FIELD_RADIUS_M (no field size chosen yet)"

    # What geo/vegetation.py and geo/thermal.py fall back to when the caller
    # passes radius_m=None. Both read ``settings.ee_neighbourhood_m or
    # DEFAULT_RADIUS_M`` -- not DEFAULT_RADIUS_M -- so quoting the module
    # constant alone would misreport a configured neighbourhood.
    veg_fallback = float(settings.ee_neighbourhood_m or _vegetation.DEFAULT_RADIUS_M)
    thermal_fallback = float(settings.ee_neighbourhood_m or _thermal.DEFAULT_RADIUS_M)

    veg = {
        "leg": "vegetation",
        "radius_m": field_r,
        "radius_source": field_source,
        "pixel_m": float(_vegetation.S2_RESOLUTION_M),
        "reason": (
            f"Sentinel-2 is {_vegetation.S2_RESOLUTION_M:g} m, so the field radius "
            "is used as given; one pixel is smaller than any real field"
        ),
        "assets": [_vegetation.S2_ASSET],
        "default_when_unset_m": veg_fallback,
    }
    thermal = {
        "leg": "thermal",
        "radius_m": field_r,
        "radius_source": field_source,
        "pixel_m": float(_thermal.LST_RESOLUTION_M),
        "reason": (
            f"Landsat surface temperature is served on the {_thermal.LST_RESOLUTION_M:g} m "
            "L2 grid (100 m native)"
        ),
        "assets": list(_thermal.LANDSAT_COLLECTIONS),
        "default_when_unset_m": thermal_fallback,
    }
    water = {
        "leg": "water",
        "radius_m": float(_water.DEFAULT_RADIUS_M),
        "radius_source": "geo/water.py DEFAULT_RADIUS_M; the field radius is not used here",
        "pixel_m": None,
        "reason": (
            "SMAP is an 11 km product and ESI 5.5 km: a point sample of either is "
            "a number about nothing, so this leg reduces over its own "
            f"{_water.DEFAULT_RADIUS_M:g} m footprint regardless of the field radius"
        ),
        "assets": [_water.SSEBOP_DEKADAL, _water.ESI_4WK, _water.SMAP_L4, _water.CHIRPS_DAILY],
        "default_when_unset_m": float(_water.DEFAULT_RADIUS_M),
    }
    soil = {
        "leg": "soil",
        "radius_m": None,
        "radius_source": "no buffer: the point itself",
        "pixel_m": None,
        "reason": "soil is read at the exact pixel; the chain is iSDA -> POLARIS -> SoilGrids",
        "assets": [
            "ISDASOIL/Africa/v1",
            "projects/sat-io/open-datasets/polaris",
            "projects/soilgrids-isric",
        ],
        "default_when_unset_m": None,
    }
    context = {
        "leg": "context",
        # Two reads, two footprints: one number here would be a lie about one of
        # them, so the leg names both and leaves the single field blank.
        "radius_m": None,
        "radius_source": "two reads; see 'reads'",
        "reads": [
            {
                "quantity": "is_cropland",
                "radius_m": None,
                "radius_source": "no buffer: the point itself",
            },
            {
                "quantity": "irrigation_regime",
                "radius_m": float(_context.IRRIGATION_NEIGHBOURHOOD_M),
                "radius_source": "geo/context.py IRRIGATION_NEIGHBOURHOOD_M",
            },
        ],
        "pixel_m": None,
        "reason": (
            "cropland extent is read at the pixel; the irrigation-regime read "
            f"widens to {_context.IRRIGATION_NEIGHBOURHOOD_M:g} m inside geo/context.py"
        ),
        "assets": [
            _context.DEAF_CROPLAND_PROB,
            _context.GFSAD_GCEP30,
            _context.ESA_WORLDCOVER,
        ],
        "default_when_unset_m": float(_context.IRRIGATION_NEIGHBOURHOOD_M),
    }
    suitability = {
        "leg": "suitability",
        "radius_m": float(settings.ee_neighbourhood_m),
        "radius_source": (
            "CROPUP_EE_NEIGHBOURHOOD_M; crop_selection is run without a radius, "
            "so the field radius is not used here either"
        ),
        "pixel_m": float(_suitability.NATIVE_SCALE_M),
        "reason": (
            "CropSuite masks single pixels -- at Arusha 47 of 48 crops in all 6 "
            "scenarios -- so a neighbourhood reduction is used and reported "
            "(SPEC 3.3)"
        ),
        "assets": [_suitability.CROP_SUITABILITY, _suitability.CLIMATE_SUITABILITY],
        "default_when_unset_m": float(settings.ee_neighbourhood_m),
    }
    climate = {
        "leg": "climate",
        "radius_m": None,
        "radius_source": "no buffer: the point itself",
        "pixel_m": None,
        "reason": "Köppen is computed in Earth Engine from climate normals at the pixel",
        "assets": ["WORLDCLIM/V1/MONTHLY", "IDAHO_EPSCOR/TERRACLIMATE"],
        "default_when_unset_m": None,
    }

    by_analysis = {
        "plant_health": [veg, thermal, soil, water, context],
        "irrigation": [water, soil, veg, thermal],
        "crop_selection": [suitability, climate, soil, context],
    }
    if analysis in by_analysis:
        return by_analysis[analysis]
    # No analysis chosen yet: name every footprint the app can use, so the map
    # can show what a run would cover once the farmer picks a question.
    return [veg, thermal, soil, water, context, suitability, climate]


def field_report(
    bag: SlotBag,
    *,
    analysis: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """The ``GET /api/geo/field`` payload.

    ``resolved`` is ``False`` -- with ``reason`` naming what is missing -- until
    the bag holds a location that carries coordinates. A place name with no
    point is not a field, and this endpoint says so rather than drawing a circle
    somewhere plausible.
    """
    settings = settings or get_settings()
    ref: Mapping[str, Any] | None = bag.field_ref()
    location = bag.get("location")
    # One radius for the whole report. The footprints and the request block are
    # two views of the same buffer and must never quote different numbers.
    effective_radius, effective_source = effective_field_radius_m(bag, settings)

    if ref is None:
        if location is None:
            reason = "no location has been given yet"
        else:
            reason = (
                f"{location.label!r} was never resolved to coordinates, and no "
                "polygon is drawn for a place that cannot be put on the map"
            )
        return {
            "resolved": False,
            "reason": reason,
            "location": location.as_dict() if location is not None else None,
            "field": None,
            "request": None,
            "footprints": reduction_footprints(
                analysis, settings, field_radius_m=effective_radius
            ),
            "analysis": analysis,
        }

    radius = float(ref.get("radius_m") or effective_radius)
    radius_source = "session" if ref.get("radius_m") else effective_source
    lat, lon = float(ref["lat"]), float(ref["lon"])
    return {
        "resolved": True,
        "reason": None,
        "location": location.as_dict() if location is not None else None,
        "label": ref.get("label"),
        "origin": ref.get("origin"),
        "source": ref.get("source"),
        "confirmed": bool(ref.get("confirmed")),
        # What is literally sent. Not approximate, not derived.
        "request": {
            "expression": "ee.Geometry.Point([lon, lat]).buffer(radius_m)",
            "lat": lat,
            "lon": lon,
            "radius_m": radius,
            "radius_source": radius_source,
            "crs": "EPSG:4326",
        },
        "field": field_polygon(lat, lon, radius),
        "centre": {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"role": "centre", "label": ref.get("label")},
        },
        "footprints": reduction_footprints(analysis, settings, field_radius_m=radius),
        "analysis": analysis,
        "note": (
            "the three numbers under 'request' are what Earth Engine receives for "
            "the legs that take the field radius; 'field' is a locally drawn ring "
            "of that disc, and 'footprints' names every reduction the chosen "
            "analysis performs -- each with its own radius and the thing that "
            "decides it, because the legs do not share one"
        ),
    }
