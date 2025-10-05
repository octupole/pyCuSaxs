// MathJax configuration for MkDocs Material
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',
    tagSide: 'right',
    tagIndent: '.8em',
    packages: {'[+]': ['ams', 'color', 'physics']},
    macros: {
      // Common vectors
      RR: "{\\mathbb{R}}",
      ZZ: "{\\mathbb{Z}}",
      NN: "{\\mathbb{N}}",
      CC: "{\\mathbb{C}}",

      // Bold vectors
      vb: ["{\\mathbf{#1}}", 1],
      vr: ["{\\mathbf{r}_{#1}}", 1],
      vq: ["{\\mathbf{q}_{#1}}", 1],

      // Common operators
      grad: "{\\nabla}",
      div: "{\\nabla \\cdot}",
      curl: "{\\nabla \\times}",

      // SAXS-specific
      Iq: "I(q)",
      Fq: "F(\\mathbf{q})",
      fq: "f(q)",

      // Units
      angstrom: "{\\mathrm{\\AA}}",
      inv: "^{-1}",

      // Averages
      avg: ["{\\langle #1 \\rangle}", 1],

      // Complex conjugate
      conj: ["{{#1}^*}", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams', '[tex]/color', '[tex]/physics']
  },
  svg: {
    fontCache: 'global'
  },
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        console.log('MathJax initial typesetting complete');
      });
    }
  }
};

// Load MathJax from CDN
document$.subscribe(() => {
  MathJax.typesetPromise()
})
