// Minimal DOM stub: runs the report's own render block against the real payload.
const fs = require("fs");
const path = process.argv[2];
const html = fs.readFileSync(path, "utf8");
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (blocks.length !== 2) throw new Error(`esperava 2 blocos de script, achei ${blocks.length}`);

class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this._text = "";
    this.className = "";
    this.dataset = {};
    this.style = {};
    this.classList = {
      add: c => { this.className = `${this.className} ${c}`.trim(); },
      remove: c => { this.className = this.className.split(/\s+/).filter(x => x && x !== c).join(" "); },
    };
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() {
    return this.children.length ? this.children.map(c => c.textContent).join("") : this._text;
  }
  append(...nodes) { for (const n of nodes) this.children.push(n); }
  replaceChildren(...nodes) { this.children = []; this._text = ""; this.append(...nodes); }
  get firstChild() { return this.children[0]; }
  closest() { return new Node("div"); }
}
class TextNode {
  constructor(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text; }
}

const byId = new Map();
for (const id of html.matchAll(/id="([a-z0-9-]+)"/g)) byId.set(id[1], new Node("div"));

const document = {
  getElementById: id => {
    if (!byId.has(id)) throw new Error(`getElementById("${id}") retornou null`);
    return byId.get(id);
  },
  querySelector: sel => {
    const id = sel.replace(/^#/, "");
    if (!byId.has(id)) throw new Error(`querySelector("${sel}") retornou null`);
    return byId.get(id);
  },
  createElement: tag => new Node(tag),
  createTextNode: t => new TextNode(t),
};
const frames = [];
const context = {
  document,
  console: { assert: (ok, msg) => { if (!ok) throw new Error(`console.assert: ${msg}`); } },
  requestAnimationFrame: fn => frames.push(fn),
};
const vm = require("node:vm");
vm.createContext(context);
// Os dois blocos contêm o payload e a renderização do relatório.
for (const index of [0, 1]) {
  vm.runInContext(blocks[index], context, { filename: `block-${index}.js` });
}
for (const fn of frames) fn();

// `const REPORT` é binding lexical do script, não propriedade do contexto:
// só é alcançável avaliando o identificador dentro do próprio contexto.
const payload = vm.runInContext("REPORT", context);

const text = id => byId.get(id).textContent;
const checks = [
  ["dataset-size", () => /^[\d.]+$/.test(text("dataset-size"))],
  ["unique-size", () => /^[\d.]+$/.test(text("unique-size"))],
  ["class-balance", () => text("class-balance").includes("positivas")],
  ["class-balance-bars", () => byId.get("class-balance-bars").children.length === 2],
  ["class-balance-bars widths", () =>
    byId.get("class-balance-bars").children.every(c => /%$/.test(c.style.width))],
  ["split-sizes", () => text("split-sizes").includes("treino")],
  ["split-sizes-bars", () => byId.get("split-sizes-bars").children.length === 3],
  ["quality-stats", () => byId.get("quality-stats").children.length === 8],
  ["eda-stats", () => byId.get("eda-stats").children.length === 13],
  ["eda-samples", () => byId.get("eda-samples").children.length === 6],
  ["model-comparison", () => byId.get("model-comparison").children.length === 1],
  ["cross-validation", () => byId.get("cross-validation").children.length === 7],
  ["confusion-matrices", () => byId.get("confusion-matrices").children.length === 2],
  ["error-examples", () => Array.isArray(byId.get("error-examples").children)],
  // Contagens derivadas do payload, para não quebrarem a cada decisão nova.
  ["decision-list", () =>
    byId.get("decision-list").children.length === payload.decisions.length],
  ["report-meta", () => byId.get("report-meta").children.length === 5],
  ["normalization-comparison", () =>
    byId.get("normalization-comparison").children.length === 1],
  ["normalization-verdict", () => byId.get("normalization-verdict").textContent.length > 0],
];
let failed = 0;
for (const [name, check] of checks) {
  let ok = false, error = "";
  try { ok = check(); } catch (e) { error = ` (${e.message})`; }
  if (!ok) { failed += 1; console.log(`FALHOU  ${name}${error}`); }
  else console.log(`ok      ${name}`);
}
console.log(failed ? `\n${failed} verificações falharam` : "\ntodas as verificações passaram");
process.exit(failed ? 1 : 0);
