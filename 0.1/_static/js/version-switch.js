/*
 * The version menu and banner of NiFreeze's documentation.
 *
 * Every published version runs this file (the deploy copies it into each one), so it relies
 * only on what all of them have: Sphinx's ``data-content_root`` on ``<html>``, and
 * ``versions.json`` in the folder above the version's own. The menu goes into
 * ``#nf-versions`` (``_templates/sidebar/versions.html``), or into ``#v-tags`` in builds
 * made before that template.
 */
(function () {
  "use strict";

  var KIND_LABELS = { dev: "development", prerelease: "pre-release" };

  // The folder of this version (e.g., ``.../nifreeze/0.2/``), and the site's root above it
  var versionRoot = new URL(
    document.documentElement.dataset.content_root || "./",
    window.location.href,
  );
  var siteRoot = new URL("../", versionRoot);
  var currentPath = decodeURIComponent(
    versionRoot.pathname.slice(siteRoot.pathname.length).replace(/\/$/, ""),
  );
  // The page within the version, so that switching keeps it
  var page = window.location.pathname.startsWith(versionRoot.pathname)
    ? window.location.pathname.slice(versionRoot.pathname.length)
    : "";

  function compareVersions(a, b) {
    // Numeric comparison of the release segments (``26.10`` after ``26.9``)
    var pa = String(a).split(/[^0-9]+/).filter(Boolean).map(Number);
    var pb = String(b).split(/[^0-9]+/).filter(Boolean).map(Number);
    for (var i = 0; i < Math.max(pa.length, pb.length); i++) {
      var d = (pa[i] || 0) - (pb[i] || 0);
      if (d !== 0) return d;
    }
    return 0;
  }

  // Both schemas: ``{"latest": ..., "versions": [...]}``, and the older
  // ``{"tags": [...], "heads": [...]}``
  function normalize(data) {
    if (Array.isArray(data.versions)) {
      return { latest: data.latest || null, versions: data.versions };
    }
    var tags = (data.tags || []).slice().sort(compareVersions).reverse();
    var versions = (data.heads || []).map(function (head) {
      return { name: head, path: head, kind: "dev" };
    });
    tags.forEach(function (tag) {
      versions.push({ name: tag, path: tag, kind: "release" });
    });
    return { latest: tags.length ? tags[0] : null, versions: versions };
  }

  function label(entry, latest) {
    // The development version goes by its folder (main); the others by their version
    var name = entry.kind === "dev" ? entry.name : entry.version || entry.name;
    if (entry.path === latest) return name + " (latest release)";
    if (KIND_LABELS[entry.kind]) return name + " (" + KIND_LABELS[entry.kind] + ")";
    return name;
  }

  function urlIn(path, target) {
    return new URL(encodeURI(path) + "/" + target, siteRoot).href;
  }

  // Go to the same page in another version, or to its index when it lacks the page
  function go(path, event) {
    if (event) event.preventDefault();
    var samePage = urlIn(path, page + window.location.hash);
    if (!page) {
      window.location.assign(samePage);
      return;
    }
    fetch(urlIn(path, page), { method: "HEAD" })
      .then(function (response) {
        window.location.assign(response.ok ? samePage : urlIn(path, "index.html"));
      })
      .catch(function () {
        window.location.assign(samePage);
      });
  }

  function link(entry, text, className) {
    var a = document.createElement("a");
    a.href = urlIn(entry.path, page);
    a.textContent = text;
    if (className) a.className = className;
    a.addEventListener("click", function (event) {
      go(entry.path, event);
    });
    return a;
  }

  function fillMenu(info) {
    var current = null;
    info.versions.forEach(function (entry) {
      if (entry.path === currentPath) current = entry;
    });

    var menu = document.getElementById("nf-versions");
    var list = menu ? menu.querySelector("ul") : document.getElementById("v-tags");
    if (!list) return current;

    info.versions.forEach(function (entry) {
      var item = document.createElement("li");
      item.className = "toctree-l2";
      if (entry.path === currentPath) item.className += " current";
      var a = link(entry, label(entry, info.latest), "reference internal");
      // Builds made before ``versions.html`` have no style for the menu
      if (!menu) a.style.textDecoration = "none";
      item.appendChild(a);
      list.appendChild(item);
    });

    if (menu && current) menu.querySelector("summary").textContent = label(current, info.latest);
    // Builds made before ``versions.html`` printed the version twice
    var slug = document.getElementById("version-slug");
    if (slug) slug.remove();
    return current;
  }

  function addBanner(info, current) {
    if (!current || !info.latest || current.path === info.latest) return;
    var latest = info.versions.filter(function (entry) {
      return entry.path === info.latest;
    })[0];
    if (!latest) return;

    var what = {
      dev: "the development version",
      prerelease: "a pre-release, " + (current.version || current.name),
    }[current.kind] || "an older release, " + (current.version || current.name);

    // Furo's announcement bar, which every build's stylesheet has, wrapping on phones
    var content = document.createElement("aside");
    content.className = "announcement-content";
    content.style.whiteSpace = "normal";
    content.appendChild(document.createTextNode("This is the documentation of " + what + ". "));
    content.appendChild(document.createTextNode("The latest release is "));
    content.appendChild(link(latest, latest.version || latest.name));
    content.appendChild(document.createTextNode("."));

    var banner = document.createElement("div");
    banner.className = "announcement nf-version-banner";
    banner.style.height = "auto";
    banner.style.minHeight = "var(--header-height)";
    banner.appendChild(content);

    var page = document.querySelector(".page");
    if (page) page.parentNode.insertBefore(banner, page);
  }

  function start() {
    fetch(new URL("versions.json", siteRoot).href)
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.json();
      })
      .then(function (data) {
        var info = normalize(data);
        addBanner(info, fillMenu(info));
      })
      .catch(function () {
        // No versions.json (e.g., a local build): show this version without a menu
        var menu = document.getElementById("nf-versions");
        if (menu) {
          var current = document.createElement("span");
          current.className = "nf-versions-current";
          current.textContent = menu.querySelector("summary").textContent;
          menu.replaceChildren(current);
        }
        var oldMenu = document.getElementById("version-menu");
        if (oldMenu) oldMenu.hidden = true;
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
