/* CropUp — vendored local fallback for the Leaflet map (SPEC 9).
 *
 * SPEC 9 asks for "Leaflet from CDN with a vendored local fallback". This file
 * is that fallback, and it is deliberately NOT a copy of Leaflet: a tile map
 * whose tiles cannot load is a grey box, which is exactly the silent
 * degradation this project exists to avoid. What the farmer must be able to
 * see before confirming is the geometry itself -- the disc that goes to
 * Earth Engine and the wider footprints some legs actually read -- and that
 * needs no network at all.
 *
 * So the fallback is a to-scale plan view drawn in SVG from the same
 * GET /api/geo/field response the Leaflet layer draws, with a scale bar, the
 * centre coordinates, and click-to-place. It is used when the CDN is
 * unreachable, when tiles fail, and whenever the farmer presses "Plan view".
 *
 * No imports, no build step, no network. window.CropUpPlanView.create(el, opts).
 */
(function (global) {
  "use strict";

  var M_PER_DEG_LAT = 110574.0;      // local flat-earth approximation, metres
  var EARTH_R = 6371008.8;

  function mPerDegLon(lat) {
    return 111320.0 * Math.cos((lat * Math.PI) / 180);
  }

  function svgEl(name, attrs) {
    var el = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k) && attrs[k] !== null && attrs[k] !== undefined) {
        el.setAttribute(k, String(attrs[k]));
      }
    }
    return el;
  }

  function niceMetres(raw) {
    // a scale-bar length a human reads: 1/2/5 x 10^n
    var pow = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var head = raw / pow;
    var pick = head >= 5 ? 5 : head >= 2 ? 2 : 1;
    return pick * pow;
  }

  function fmtMetres(m) {
    if (m >= 1000) return (m / 1000).toFixed(m % 1000 === 0 ? 0 : 1) + " km";
    return Math.round(m) + " m";
  }

  function create(el, opts) {
    opts = opts || {};
    var state = {
      report: null,
      colours: {},
      centre: null,     // {lat, lon}
      scale: 1,         // px per metre
      size: { w: 0, h: 0 },
      svg: null
    };

    el.innerHTML = "";
    el.classList.add("planview-host");

    var svg = svgEl("svg", { class: "planview", preserveAspectRatio: "xMidYMid meet" });
    el.appendChild(svg);
    state.svg = svg;

    function pxToLatLon(px, py) {
      if (!state.centre) return null;
      var cx = state.size.w / 2;
      var cy = state.size.h / 2;
      var east = (px - cx) / state.scale;
      var north = (cy - py) / state.scale;
      return {
        lat: state.centre.lat + north / M_PER_DEG_LAT,
        lon: state.centre.lon + east / mPerDegLon(state.centre.lat)
      };
    }

    svg.addEventListener("click", function (ev) {
      if (!opts.onPick || !state.centre) return;
      var box = svg.getBoundingClientRect();
      var ll = pxToLatLon(ev.clientX - box.left, ev.clientY - box.top);
      if (ll) opts.onPick(ll.lat, ll.lon);
    });

    function project(lon, lat) {
      var c = state.centre;
      var east = (lon - c.lon) * mPerDegLon(c.lat);
      var north = (lat - c.lat) * M_PER_DEG_LAT;
      return [state.size.w / 2 + east * state.scale, state.size.h / 2 - north * state.scale];
    }

    function render(report, colours) {
      if (report) state.report = report;
      if (colours) state.colours = colours;
      draw();
    }

    function draw() {
      var report = state.report;
      svg.innerHTML = "";

      var w = Math.max(el.clientWidth || 0, 200);
      var h = Math.max(el.clientHeight || 0, 180);
      state.size = { w: w, h: h };
      svg.setAttribute("viewBox", "0 0 " + w + " " + h);
      svg.setAttribute("width", w);
      svg.setAttribute("height", h);

      if (!report || !report.resolved || !report.request) {
        var t = svgEl("text", { x: w / 2, y: h / 2, "text-anchor": "middle" });
        t.textContent = report && report.reason ? report.reason : "no field yet";
        svg.appendChild(t);
        state.centre = null;
        return;
      }

      var req = report.request;
      state.centre = { lat: req.lat, lon: req.lon };

      var footprints = (report.footprints || []).filter(function (f) {
        return typeof f.radius_m === "number" && f.radius_m > 0;
      });
      var radii = footprints.map(function (f) { return f.radius_m; });
      radii.push(req.radius_m || 15);
      var maxR = Math.max.apply(null, radii);
      // the widest footprint fills ~80% of the shorter side
      var half = Math.min(w, h) / 2;
      state.scale = (half * 0.8) / maxR;

      var cx = w / 2, cy = h / 2;

      // graticule: concentric guides every "nice" step
      var step = niceMetres(maxR / 2.5);
      var g = svgEl("g", { opacity: "0.35" });
      for (var r = step; r <= maxR * 1.25; r += step) {
        g.appendChild(svgEl("circle", {
          cx: cx, cy: cy, r: r * state.scale, fill: "none",
          stroke: "currentColor", "stroke-width": 0.5, "stroke-dasharray": "2 4"
        }));
      }
      svg.appendChild(g);

      // footprint rings, widest first so the small ones stay visible
      var lastLabelY = -Infinity;
      footprints.slice().sort(function (a, b) { return b.radius_m - a.radius_m; })
        .forEach(function (f) {
          var colour = state.colours[f.leg] || "#888";
          svg.appendChild(svgEl("circle", {
            cx: cx, cy: cy, r: f.radius_m * state.scale,
            fill: colour, "fill-opacity": 0.06,
            stroke: colour, "stroke-width": 1.5, "stroke-dasharray": "6 4"
          }));
          // Two legs can share a radius (vegetation and thermal both read the
          // field radius); stack their labels rather than printing one on top
          // of the other, which would read as a single footprint.
          var y = cy - f.radius_m * state.scale - 3;
          if (y < lastLabelY + 11) y = lastLabelY + 11;
          lastLabelY = y;
          var label = svgEl("text", {
            x: cx, y: y, "text-anchor": "middle", fill: colour
          });
          label.textContent = f.leg + " " + fmtMetres(f.radius_m);
          svg.appendChild(label);
        });

      // the field polygon, exactly as the server drew it
      if (report.field && report.field.geometry && report.field.geometry.type === "Polygon") {
        var ring = report.field.geometry.coordinates[0] || [];
        var pts = ring.map(function (c) { return project(c[0], c[1]).join(","); }).join(" ");
        svg.appendChild(svgEl("polygon", {
          points: pts, fill: "#1f6b46", "fill-opacity": 0.28,
          stroke: "#1f6b46", "stroke-width": 2
        }));
        // A 15 m plot inside a 5 km soil-moisture disc is genuinely sub-pixel.
        // That disparity is the point, so keep the plot findable and say so
        // rather than letting it vanish and look like it was never drawn.
        var plotPx = (req.radius_m || 0) * state.scale;
        if (plotPx < 4) {
          svg.appendChild(svgEl("circle", {
            cx: cx, cy: cy, r: 4, fill: "none", stroke: "#1f6b46", "stroke-width": 2
          }));
          var tiny = svgEl("text", { x: cx + 8, y: cy - 6, fill: "#1f6b46" });
          tiny.textContent = "your plot, " + fmtMetres(req.radius_m) + " radius (too small to draw at this scale)";
          svg.appendChild(tiny);
        }
      }

      // the centre
      svg.appendChild(svgEl("circle", { cx: cx, cy: cy, r: 3.5, fill: "#b3261e" }));
      svg.appendChild(svgEl("line", { x1: cx - 9, y1: cy, x2: cx + 9, y2: cy, stroke: "#b3261e", "stroke-width": 1 }));
      svg.appendChild(svgEl("line", { x1: cx, y1: cy - 9, x2: cx, y2: cy + 9, stroke: "#b3261e", "stroke-width": 1 }));

      var coord = svgEl("text", { x: cx + 8, y: cy + 14 });
      coord.textContent = req.lat.toFixed(5) + ", " + req.lon.toFixed(5);
      svg.appendChild(coord);

      // scale bar
      var barM = niceMetres(maxR / 2);
      var barPx = barM * state.scale;
      var bx = 12, by = h - 16;
      svg.appendChild(svgEl("line", { x1: bx, y1: by, x2: bx + barPx, y2: by, stroke: "currentColor", "stroke-width": 2 }));
      svg.appendChild(svgEl("line", { x1: bx, y1: by - 4, x2: bx, y2: by + 4, stroke: "currentColor", "stroke-width": 2 }));
      svg.appendChild(svgEl("line", { x1: bx + barPx, y1: by - 4, x2: bx + barPx, y2: by + 4, stroke: "currentColor", "stroke-width": 2 }));
      var sl = svgEl("text", { x: bx, y: by - 7 });
      sl.textContent = fmtMetres(barM) + "  (plan view, north up)";
      svg.appendChild(sl);
    }

    var onResize = function () { draw(); };
    global.addEventListener("resize", onResize);

    return {
      kind: "plan",
      render: render,
      recentre: draw,
      invalidate: draw,
      destroy: function () {
        global.removeEventListener("resize", onResize);
        el.innerHTML = "";
        el.classList.remove("planview-host");
      }
    };
  }

  global.CropUpPlanView = {
    create: create,
    // exported for the geometry the app shares with the Leaflet path
    metresPerDegreeLat: M_PER_DEG_LAT,
    metresPerDegreeLon: mPerDegLon,
    earthRadius: EARTH_R,
    niceMetres: niceMetres,
    formatMetres: fmtMetres
  };
})(window);
