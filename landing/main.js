// Noelle website: download links from config.js, light/dark toggle, screenshot swap, copy buttons, mobile menu.
(function () {
  "use strict";

  var root = document.documentElement;
  var links = window.NOELLE_LINKS || {};

  // Every element with data-link="download|files|source" points at the URL from config.js.
  document.querySelectorAll("[data-link]").forEach(function (anchor) {
    var url = links[anchor.getAttribute("data-link")];
    if (url && url.indexOf("YOUR-PROJECT") === -1) {
      anchor.setAttribute("href", url);
      if (/^https?:/.test(url)) anchor.setAttribute("rel", "noopener");
    }
  });

  function currentTheme() {
    var forced = root.getAttribute("data-theme");
    if (forced) return forced;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function syncScreenshots() {
    var theme = currentTheme();
    document.querySelectorAll("img[data-light][data-dark]").forEach(function (img) {
      var src = img.getAttribute(theme === "dark" ? "data-dark" : "data-light");
      if (img.getAttribute("src") !== src) img.setAttribute("src", src);
    });
  }

  document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
    button.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("noelle-theme", next); } catch (e) { /* private mode: not remembered */ }
      syncScreenshots();
    });
  });
  if (window.matchMedia) {
    var query = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function () { if (!root.getAttribute("data-theme")) syncScreenshots(); };
    if (query.addEventListener) query.addEventListener("change", onChange);
  }
  syncScreenshots();

  // Copy buttons: data-copy="<id of the <pre>>"
  document.querySelectorAll("[data-copy]").forEach(function (button) {
    button.addEventListener("click", function () {
      var source = document.getElementById(button.getAttribute("data-copy"));
      if (!source || !navigator.clipboard) return;
      navigator.clipboard.writeText(source.innerText).then(function () {
        button.classList.add("done");
        setTimeout(function () { button.classList.remove("done"); }, 1400);
      });
    });
  });

  // Mobile menu
  var menu = document.querySelector("[data-menu]");
  var nav = document.querySelector(".nav-links");
  if (menu && nav) {
    menu.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      menu.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.addEventListener("click", function (event) {
      if (event.target.closest("a")) { nav.classList.remove("open"); menu.setAttribute("aria-expanded", "false"); }
    });
  }

  document.querySelectorAll("[data-year]").forEach(function (el) { el.textContent = String(new Date().getFullYear()); });
})();
