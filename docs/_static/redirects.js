// Will match any urls containing the strings
// - "/dev/"
// - "/stable/"
// - "/X.Y.Z/"
// where X.Y.Z is a version string and redirect the page to its stable version.

let pattern = /\/dev\/|\/stable\/|\/\d+.\d+.\d+\//;

if (!pattern.test(window.location.href.toLowerCase())) {
   window.location.href = 'http://napari.org/stable'+window.location.pathname;
}
