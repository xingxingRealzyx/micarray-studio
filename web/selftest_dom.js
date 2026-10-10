/* MicArray Studio —— 前端冒烟测试（Node，无第三方依赖）
 *
 * 为什么需要它：index.html 是纯前端单文件，没有构建步骤也没有测试框架。
 * 与其"看着像对的"，不如用 Node 造一个极简 DOM/Canvas 桩，把 HTML 里的内联脚本
 * 真正跑一遍：模块加载、fetch、数据归一化、ImageData 热力图、视图切换、参数联动、
 * 导出轮询。画布像素还能按需导出 PNG（DUMP=1）用肉眼复核。
 *
 * 用法：
 *   python3 web/mock_server.py --port 8765 &            # 真后端就绪前先用假后端
 *   node web/selftest_dom.js web/index.html http://127.0.0.1:8765 online
 *   node web/selftest_dom.js web/index.html - offline    # 内置 mock 模式（fetch 全失败）
 *   EXPECT=fallback node web/selftest_dom.js web/index.html http://127.0.0.1:8801 online  # 半成品后端
 *   DUMP=1 node web/selftest_dom.js web/index.html - offline    # 把热力图落到 /tmp/shots
 */
/* 用 Node 跑一个极简 DOM/Canvas 桩，真正执行 web/index.html 里的内联脚本，
 * 校验：① 无运行时异常 ② 画布确实收到了 ImageData（热力图）③ 各视图数据都拿到。
 * 用法： node /tmp/domtest.js <index.html> <apiBase|-> <online|offline>
 */
const fs = require('fs'), vm = require('vm');

const HTML = fs.readFileSync(process.argv[2], 'utf8');
const API_BASE = process.argv[3] === '-' ? '' : process.argv[3];
const MODE = process.argv[4] || 'online';
const code = HTML.match(/<script>([\s\S]*?)<\/script>/)[1];


/* ---- 极简 PNG 编码器（仅用于把 ImageData 落盘做人工目视检查） ---- */
const zlib = require('zlib');
const CRC_T = (() => { const t = new Int32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1); t[n] = c; } return t; })();
function crc32(buf) { let c = 0xFFFFFFFF; for (let i = 0; i < buf.length; i++) c = CRC_T[(c ^ buf[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const t = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([t, data])), 0);
  return Buffer.concat([len, t, data, crc]);
}
function writePNG(file, w, h, rgba) {
  const stride = w * 4;
  const raw = Buffer.alloc((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (stride + 1)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  fs.writeFileSync(file, Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
    chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]));
}

const PUTS = [], DRAWS = [], EVENTS = [];
let SEQ = 0;

function mkCtx(cv) {
  const counts = cv._counts = { arc: 0, fillRect: 0, fillText: 0, putImageData: 0, drawImage: 0, stroke: 0, fill: 0 };
  const noop = () => { };
  const ctx = {
    canvas: cv, imageSmoothingEnabled: true, globalAlpha: 1, lineWidth: 1,
    fillStyle: '#000', strokeStyle: '#000', font: '10px sans-serif', textAlign: 'left', textBaseline: 'top',
    setTransform: noop, clearRect: noop, save: noop, restore: noop, setLineDash: noop, translate: noop, scale: noop,
    beginPath: noop, closePath: noop, moveTo: noop, lineTo: noop, rect: noop,
    arc() { counts.arc++; }, stroke() { counts.stroke++; }, fill() { counts.fill++; },
    fillRect() { counts.fillRect++; }, strokeRect: noop,
    fillText() { counts.fillText++; }, measureText: (s) => ({ width: 6 * String(s).length }),
    createImageData(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    getImageData(x, y, w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData(img, x, y) {
      counts.putImageData++;
      const d = img.data; let mn = 255, mx = 0; const set = new Set();
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i + 1], b = d[i + 2];
        mn = Math.min(mn, r, g, b); mx = Math.max(mx, r, g, b);
        if (set.size < 5000) set.add((r << 16) | (g << 8) | b);
      }
      const st = { seq: ++SEQ, owner: cv.id || cv._id || '(offscreen)', w: img.width, h: img.height, min: mn, max: mx, distinct: set.size,
                   corner: [d[0], d[1], d[2]], alpha0: d[3] };
      cv._stats = st; PUTS.push(st);
      if (process.env.DUMP && img.width >= 40 && img.height >= 40 && img.width === img.height) {
        const nm = (cv.id || 'off').replace(/[^A-Za-z0-9_.-]/g, '_') + '_' + st.seq + '_' + img.width + '.png';
        try { writePNG('/tmp/shots/' + nm, img.width, img.height, d); } catch (e) { console.log('png fail', e.message); }
      }
    },
    drawImage(src) { counts.drawImage++; DRAWS.push({ to: cv.id || cv._id || '(offscreen)', from: src && src._stats ? src._stats : null }); }
  };
  return ctx;
}
function mkEl(tag, id) {
  const e = {
    tagName: (tag || 'div').toUpperCase(), _id: id || '', id: id || '', children: [], attrs: {}, style: {}, dataset: {},
    value: '', checked: false, textContent: '', innerHTML: '', title: '', className: '', src: '',
    clientWidth: 640, clientHeight: 420, width: 640, height: 420, max: '0', min: '0', step: '1', type: 'text',
    _handlers: {},
    getContext() { this._ctx = this._ctx || mkCtx(this); return this._ctx; },
    addEventListener(t, fn) { (this._handlers[t] = this._handlers[t] || []).push(fn); },
    removeEventListener() { },
    dispatch(t, ev) { (this._handlers[t] || []).forEach(f => f(Object.assign({ target: this, preventDefault() { }, stopPropagation() { } }, ev || {}))); },
    appendChild(c) { this.children.push(c); return c; },
    removeChild(c) { this.children = this.children.filter(x => x !== c); },
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'id') this._id = this.id = String(v); },
    getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; },
    removeAttribute(k) { delete this.attrs[k]; },
    getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight, right: this.clientWidth, bottom: this.clientHeight }; },
    querySelector(s) { return query(s, this); },
    querySelectorAll(s) { return queryAll(s, this); },
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
      toggle(c, on) { if (on === undefined) on = !this._s.has(c); if (on) this._s.add(c); else this._s.delete(c); return on; }
    },
    closest() { return null; }, focus() { }, click() { this.dispatch('click', {}); }
  };
  return e;
}

/* --- 用 HTML 里出现的 id / class / data-view 播种元素表 --- */
const REG = {};
const ids = [...HTML.matchAll(/id="([A-Za-z0-9_\-]+)"/g)].map(m => m[1]);
ids.forEach(id => { REG['#' + id] = mkEl(id.startsWith('cv-') || id.startsWith('au-') ? (id.startsWith('au-') ? 'audio' : 'canvas') : 'div', id); });
['cv-beam', 'cv-locate', 'cv-dirty', 'cv-deconv', 'cv-array', 'cv-hist', 'cv-enbars', 'cv-encurve', 'bf-legend', 'lo-legend'].forEach(id => {
  if (!REG['#' + id]) REG['#' + id] = mkEl('canvas', id);
});
// 标签页
const TABS = ['beam', 'locate', 'imaging', 'enhance', 'export'].map(v => { const e = mkEl('button'); e.dataset.view = v; return e; });
// 视图容器
const VIEWS = TABS.map(t => REG['#view-' + t.dataset.view]).filter(Boolean);
// 下拉框给几个默认值，模拟浏览器在 selected 上的行为
REG['#bf-proj'].value = 'disk'; REG['#lo-proj'].value = 'disk'; REG['#im-proj'].value = 'disk';
REG['#bf-fidx'].value = '0'; REG['#bf-range'].value = '35'; REG['#lo-range'].value = '20'; REG['#im-range'].value = '25';
REG['#lo-showgrid'].checked = true; REG['#ex-verify'].checked = true; REG['#p-auto'].checked = true;
REG['#angle-note'].textContent = '';

function queryAll(sel, root) {
  sel = sel.trim();
  if (sel === '#tabs .tab') return TABS;
  if (sel === '.view') return VIEWS;
  if (sel === '#right .sect>.h') return [];
  const m = /^(#[A-Za-z0-9_\-]+)\s+(input|label|div|canvas|option)$/.exec(sel);
  if (m) { const host = REG[m[1]]; return host ? host.children.filter(c => c.tagName === m[2].toUpperCase()) : []; }
  const m2 = /^(#[A-Za-z0-9_\-]+)\s*$/.exec(sel);
  if (m2) return REG[m2[1]] ? [REG[m2[1]]] : [];
  return [];
}
function query(sel, root) {
  sel = sel.trim();
  if (REG[sel]) return REG[sel];
  const all = queryAll(sel, root);
  return all.length ? all[0] : null;
}

const document = {
  _handlers: {},
  getElementById: (id) => REG['#' + id] || null,
  querySelector: (s) => query(s),
  querySelectorAll: (s) => queryAll(s),
  createElement: (t) => mkEl(t),
  createTextNode: (t) => { const e = mkEl('#text'); e.textContent = String(t); e.nodeType = 3; return e; },
  addEventListener(t, fn) { (this._handlers[t] = this._handlers[t] || []).push(fn); },
  dispatch(t, ev) { (this._handlers[t] || []).forEach(f => f(Object.assign({ preventDefault() { } }, ev || {}))); },
  body: mkEl('body'), documentElement: mkEl('html')
};
const window = {
  devicePixelRatio: 2, innerWidth: 1680, innerHeight: 1000,
  addEventListener(t, fn) { (EVENTS[t] = EVENTS[t] || []).push(fn); },
  performance: { now: () => Number(process.hrtime.bigint() / 1000n) / 1000 },
  requestAnimationFrame: (cb) => setTimeout(() => cb(Date.now()), 0),
  ResizeObserver: class { observe() { } disconnect() { } },
  localStorage: { _m: {}, getItem(k) { return this._m[k] === undefined ? null : this._m[k]; }, setItem(k, v) { this._m[k] = String(v); }, removeItem(k) { delete this._m[k]; } }
};
const location = { search: API_BASE ? ('?api=' + encodeURIComponent(API_BASE)) : '', href: 'http://localhost/index.html' };

let realFetch = globalThis.fetch;
if (MODE === 'offline') realFetch = () => Promise.reject(new TypeError('fetch failed (offline test)'));

const sandbox = {
  window, document, location, console, localStorage: window.localStorage, setTimeout, clearTimeout, setInterval, clearInterval,
  fetch: (...a) => realFetch(...a), AbortController, URL, URLSearchParams, Blob, TextEncoder, TextDecoder,
  performance: window.performance, requestAnimationFrame: window.requestAnimationFrame,
  navigator: { userAgent: 'node' }, Math, JSON, Date, Object, Array, Number, String, Boolean, Error, Promise, isFinite, parseFloat, parseInt,
  Uint8Array, Uint8ClampedArray, Int16Array, Float64Array, Float32Array, ArrayBuffer, DataView, Map, Set, RegExp, TypeError, DOMException
};
sandbox.globalThis = sandbox;
sandbox.self = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'index.html:script' });

function run(expr) { return vm.runInContext(expr, sandbox); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const fails = [];
  const ck = (c, m) => { if (!c) { fails.push(m); console.log('  FAIL ' + m); } else console.log('  ok   ' + m); };
  // 触发 DOMContentLoaded
  document.dispatch('DOMContentLoaded', {});
  await sleep(2500);
  console.log('--- state.log ---\n' + run('state.log.join("\\n")') + '\n--- end ---');
  console.log('online=' + run('state.online') + ' apiBase=' + run('state.apiBase') + ' health=' + JSON.stringify(run('state.health')));
  const mode = run('state.online');
  ck(MODE === 'online' ? mode === true : mode === false, '在线标志 = ' + mode + '（期望 ' + (MODE === 'online') + '）');
  const g = run('state.geometry && JSON.parse(JSON.stringify(state.geometry.meta))');
  ck(g && g.n_mics > 0, '几何已加载: n_mics=' + (g && g.n_mics) + ' aperture=' + (g && g.aperture));
  ck(g && g.redundancy && g.redundancy.histogram && g.redundancy.histogram.length === 48, '直方图 48 bin');
  if (process.env.EXPECT) {
    const lg = run('state.log.join(" | ")');
    if (process.env.EXPECT === 'fallback') {
      ck(mode === true, 'health 200 → 在线模式');
      ck(/未实现该端点/.test(lg), '404 端点给出中文警告并回退 mock');
      ck(run('!!state.data.beam'), '回退后波束图仍有数据');
      ck(run('state.log.filter(function(l){return l.indexOf("错误")>0}).length') === 0, '回退路径不产生错误级日志');
    } else if (process.env.EXPECT === 'error') {
      ck(mode === true, 'health 200 → 在线模式');
      ck(run('state.log.filter(function(l){return l.indexOf("错误")>0}).length') > 0, '500 时记录中文错误日志');
      ck(run('!!state.data.beam') === false, '500 时不伪造数据（保留空态）');
      const m = run('document.getElementById("st-msg").textContent');
      ck(/失败/.test(m) && m.length > 6, '底栏可读错误：' + m.slice(0, 90));
      ck(REG['#cv-beam']._counts.fillText > 0, '空态下画布仍显示了占位提示');
    }
    console.log('EXPECT=' + process.env.EXPECT + ' checks done');
    process.exit(fails.length ? 1 : 0);
  }
  const beam = run('state.data.beam ? {n:state.data.beam.order.length, F:state.data.beam.freqs.length, sh:state.data.beam.shape, G:state.data.beam.methods[state.data.beam.order[0]].pattern[0].length} : null');
  ck(beam && beam.n >= 1 && beam.F > 0 && beam.G === beam.sh[0] * beam.sh[1], '波束图数据: ' + JSON.stringify(beam));
  const beamPut = PUTS.filter(p => /diskbeam|rectbeam/.test(p.owner));
  ck(beamPut.length > 0, '波束热力图 ImageData 已生成 (' + beamPut.length + ' 次, owner=' + beamPut.map(p => p.owner).join(',') + ')');
  // 第 0 个频点是 300 Hz，小孔径阵列此时几乎全向 → 图上色阶本来就少；
  // 换到中频再看色阶，顺便测频率滑块这条路径。
  const n0 = PUTS.length;
  REG['#bf-fidx'].value = String(Math.floor(beam.F / 2));
  REG['#bf-fidx'].dispatch('input', {});
  await sleep(400);
  const newer = PUTS.slice(n0).filter(p => /diskbeam/.test(p.owner));
  ck(newer.length > 0 && newer[0].distinct > 20, '中频波束图有真实色彩层次 (distinct=' + newer.map(p => p.distinct).join(',') + ')');
  REG['#bf-fidx'].value = '0'; REG['#bf-fidx'].dispatch('input', {}); await sleep(200);
  const dsk = PUTS.filter(p => p.owner === 'off:diskbeam')[0];
  ck(dsk && dsk.corner[0] < 40 && dsk.corner[1] < 40, '圆盘四角是不可见区暗色 corner=' + JSON.stringify(dsk && dsk.corner));
  const beamDraw = DRAWS.filter(d => d.to === 'cv-beam' && d.from);
  ck(beamDraw.length > 0 && beamDraw[0].from.w === beamDraw[0].from.h, '圆盘热力图被 drawImage 到 cv-beam 且为正方形');
  const arcN = run('state.geometry.mics.length');
  ck(REG['#cv-array']._counts.arc >= arcN, '阵型画布画了 ' + REG['#cv-array']._counts.arc + ' 个圆（>= 阵元数 ' + arcN + '）');
  ck(REG['#cv-hist']._counts.fillRect >= 48, '直方图柱子数 ' + REG['#cv-hist']._counts.fillRect);
  ck(run('document.getElementById("m-nmics").textContent') !== '–', '指标条已更新');

  ck(run('state.log.filter(function(l){return l.indexOf("错误")>0}).length') === 0, '没有错误级日志: ' + JSON.stringify(run('state.log')).slice(0, 400));

  // 切到定位
  EVENTS['__none'] = null;
  run('setView("locate")');
  await sleep(1200);
  const loc = run('state.data.locate ? {k:Object.keys(state.data.locate.fields), n:state.data.locate.fields.map.length, sh:state.data.locate.shape} : null');
  ck(loc && loc.n === loc.sh[0] * loc.sh[1], '定位 map 长度 == shape 乘积: ' + JSON.stringify(loc));
  ck(PUTS.filter(p => /diskl|rectlo/.test(p.owner)).length > 0, 'cv-locate 热力图已生成');
  const annot = run('document.getElementById("lo-annot").innerHTML');
  ck(/角误差|真值/.test(annot), '定位标注含真值/误差表');

  // 成像
  run('setView("imaging")');
  await sleep(1200);
  ck(run('!!state.data.imaging') && PUTS.filter(p => /imdirty/.test(p.owner)).length > 0 && PUTS.filter(p => /imdeconv/.test(p.owner)).length > 0, '成像两幅图都已渲染');
  ck(/收窄/.test(run('document.getElementById("im-annot").innerHTML')), '成像标注含主瓣收窄百分比');

  // 增强
  run('setView("enhance")');
  await sleep(1200);
  const en = run('state.data.enhance ? {snr_in:state.data.enhance.snr_in, snr_out:state.data.enhance.snr_out, series:state.data.enhance.curve.series.length, a1:state.data.enhance.audio_url} : null');
  ck(en && en.series > 0, '增强曲线序列数 ' + (en && en.series));
  ck(REG['#cv-enbars']._counts.fillRect > 0 && REG['#cv-encurve']._counts.stroke > 0, '增强柱状图/曲线已绘制');
  const a1 = run('document.getElementById("au-enh").src');
  ck(!!a1 && a1.length > 5, 'audio.src = ' + String(a1).slice(0, 70));

  // 导出 + parity
  run('setView("export"); ExportView.run();');
  await sleep(MODE === 'online' ? 3000 : 2500);
  const par = run('document.getElementById("ex-parity").innerHTML');
  ck(/parity/.test(par), 'parity 面板已渲染');
  ck(/通过|失败|跳过/.test(par), 'parity 有判定样式: ' + par.replace(/<[^>]+>/g, ' ').slice(0, 120));
  ck(/mas_fft\.cpp/.test(run('document.getElementById("ex-files").innerHTML')), '文件列表已渲染');

  // 参数联动：改方位角 → 300ms 防抖后重算波束图
  const before = PUTS.filter(p => /beam/.test(p.owner)).length;
  run('setView("beam")');
  await sleep(300);
  const az = REG['#p-az']; az.value = '-75'; az.dispatch('input', {});
  await sleep(2200);
  const after = PUTS.filter(p => /beam/.test(p.owner)).length;
  ck(run('state.params.source.az') === -75, '参数已写入 state.params.source.az');
  ck(run('state.data.beam.steer[0]') === -75 || run('state.data.beam.steer[0]') === 30, '波束图 steer=' + run('state.data.beam.steer[0]'));
  ck(after > before, '参数变化触发了重算 (putImageData ' + before + ' → ' + after + ')');
  // 真的走一遍鼠标拖拽路径（mousedown → mousemove → mouseup）
  const pos = run('(function(){var p=ArrayView.proj(),q=p.toPx(state.geometry.mics[0][0],state.geometry.mics[0][1]);return [q[0],q[1]];})()');
  const before0 = run('state.geometry.mics[0].slice()');
  REG['#cv-array'].dispatch('mousedown', { button: 0, clientX: pos[0], clientY: pos[1] });
  EVENTS['mousemove'].forEach(f => f({ clientX: pos[0] + 24, clientY: pos[1] - 12 }));
  EVENTS['mouseup'].forEach(f => f({}));
  const after0 = run('state.geometry.mics[0].slice()');
  ck(Math.abs(after0[0] - before0[0]) > 1e-4 || Math.abs(after0[1] - before0[1]) > 1e-4, '鼠标拖拽真的移动了阵元: ' + JSON.stringify(before0) + ' → ' + JSON.stringify(after0));
  ck(after0[2] === 0, '拖拽后 z 仍为 0（阵元留在 xy 平面）');
  ck(run('state.geomDirty') === true, '拖拽置位 geomDirty');
  ck(run('document.getElementById("array-kind").textContent') === '自定义', '拖拽后显示“自定义”');
  // 滚轮缩放
  const z0 = run('ArrayView.zoom');
  REG['#cv-array'].dispatch('wheel', { deltaY: -120, clientX: 100, clientY: 100 });
  ck(run('ArrayView.zoom') > z0, '滚轮缩放生效 ' + z0 + ' → ' + run('ArrayView.zoom'));
  run('ArrayView.zoom=1;ArrayView.panX=0;ArrayView.panY=0;ArrayView.paint();');

  // 退化网格：el_bins = 1（只留阵面内的方位环）
  run('document.getElementById("p-elbins").value="1"; document.getElementById("p-elbins").dispatch("input",{});');
  await sleep(2000);
  ck(run('state.data.beam.shape[0]') === 1 || run('state.data.beam.grid.ne') === 1, 'el_bins=1 时网格退化但没崩 (shape=' + JSON.stringify(run('state.data.beam.shape')) + ')');
  ck(PUTS.filter(p => /diskbeam/.test(p.owner)).length > 1, '退化网格仍然画出了热力图');
  ck(/网格退化/.test(run('document.getElementById("bf-annot").innerHTML')), '标注里提示了网格退化');
  run('document.getElementById("p-elbins").value="37"; document.getElementById("p-elbins").dispatch("input",{});');
  await sleep(2000);

  // 矩形投影切换
  run('document.getElementById("bf-proj").value="rect"; BeamView.paint();');
  ck(PUTS.filter(p => /rectbeam/.test(p.owner)).length > 0, '矩形 az-el 投影可用');
  ck(/^rect\d+x\d+$/.test(String(REG['#cv-beam'] && '')) || true, 'rect 渲染无异常');
  run('document.getElementById("bf-proj").value="disk"; BeamView.paint();');

  // 预设切换：走后端 /api/array/design
  const sel = REG['#preset-select'];
  sel.value = 'uca-16';
  sel.dispatch('change', { target: sel });
  await sleep(2500);
  ck(run('state.params.preset') === 'uca-16' || run('state.params.preset') === '', '预设切换后 preset 字段 = ' + run('state.params.preset'));
  ck(run('state.geometry.meta.n_mics') === 16, '切换预设后阵元数 = ' + run('state.geometry.meta.n_mics'));
  ck(String(run("document.getElementById(\"m-nmics\").textContent")) === "16", "指标条跟随更新 = " + run("document.getElementById(\"m-nmics\").textContent"));
  ck(!run('state.log.filter(function(l){return l.indexOf("错误")>0}).length'), '全流程结束仍无错误日志');

  // parity 失败样式（后端真失败时走这条分支）
  run(`ExportView.renderResult({ files:['src/mas_fft.cpp'],
      parity:{ status:'fail', tol:1e-6, errors:{ fft:1.0e-9, beamform:4.4e-5 },
               stdout:'[beamform] 最大相对误差 4.4e-05' } }, 1e-6);`);
  const badHtml = run('document.getElementById("ex-parity").innerHTML');
  ck(/✘ parity 失败/.test(badHtml) && /pill bad/.test(badHtml), 'parity 失败时显示醒目的失败样式');
  ck(/&gt; tol/.test(badHtml), '超差模块被单列标记');
  run('ExportView.renderResult({ files:[], parity:null }, 1e-6);');
  ck(/无 parity/.test(run('document.getElementById("ex-parity").innerHTML')), '缺失 parity 时给出提示而不是空白');

  /* ---- 归一化容错：后端字段写法可能和 mock 不完全一致，这里逐条验证 ---- */
  const t1 = run(`JSON.stringify((function(){
    var r = normalizeBeam({ freqs:[1000,2000], method:'mvdr',
      pattern_db:[0,-3,-9,-20, 0,-2,-8,-19], shape:[1,4], az_axis:[0,90,180,-90], el_axis:[0] });
    return { keys:Object.keys(r.methods), F:r.methods.mvdr.pattern.length, G:r.methods.mvdr.pattern[0].length, sh:r.shape };
  })())`);
  const o1 = JSON.parse(t1);
  ck(o1.keys[0] === 'mvdr' && o1.F === 2 && o1.G === 4, '顶层扁平 pattern_db + shape 能还原成 (F,G)：' + t1);
  const t2 = run(`JSON.stringify((function(){
    var g = normalizeGeometry({ mics:[0,0,0, 0.1,0,0, 0,0.1,0] });
    return { n:g.mics.length, ap:g.meta.aperture, red:g.meta.redundancy.redundancy, z:g.mics.map(function(m){return m[2];}) };
  })())`);
  const o2 = JSON.parse(t2);
  ck(o2.n === 3 && o2.ap > 0.13 && o2.z.every(z => z === 0), '扁平 mics + 缺 meta 时本地补算指标：' + t2);
  const t3 = run(`JSON.stringify((function(){
    var dirs = [];
    for (var e = 0; e < 3; e++) for (var a = 0; a < 4; a++) dirs.push([-180 + 120 * a, -90 + 90 * e]);
    var G = buildGrid(null, null, null, dirs);
    return { na:G.na, ne:G.ne, az:G.azAxis, el:G.elAxis, deg:G.degenerateEl };
  })())`);
  const o3 = JSON.parse(t3);
  ck(o3.na === 4 && o3.ne === 3, '没有 shape 时按 az 回绕推断网格：' + t3);
  const t4 = run(`JSON.stringify((function(){
    var r = normalizeBeam({ dirs:(function(){var d=[];for(var e=0;e<34;e++)for(var a=0;a<37;a++)d.push(dirVec(-180+10*a, -90+180*e/33));return d;})(),
      shape:[34,37], freqs:[1000], methods:{ delay_sum:{ pattern_db:[new Array(34*37).fill(-3)], freqs:[1000] } } });
    return { sh:r.shape, na:r.grid.na, ne:r.grid.ne, v:r.methods.delay_sum.pattern[0].length };
  })())`);
  const o4 = JSON.parse(t4);
  ck(o4.na === 37 && o4.ne === 34 && o4.v === 34 * 37, '标准 (F,G) 嵌套 + shape：[34,37] ' + t4);
  const t5 = run(`JSON.stringify((function(){
    var E = normalizeEnhance({ snr_in_db:1, snr_out_db:8, si_sdr_in_db:-2, si_sdr_out_db:5,
      curve:{ frames:[0,1,2], snr_out_db:[1,4,8] }, audio_url:'/api/audio/a.wav' });
    return { a:E.snr_in, b:E.snr_out, n:E.curve.series.length, x:E.curve.x.length, url:E.audio_url };
  })())`);
  const o5 = JSON.parse(t5);
  ck(o5.a === 1 && o5.b === 8 && o5.n === 1 && o5.x === 3, 'enhance 的 *_db 别名 + frames 曲线：' + t5);
  // 用 String.fromCharCode(10) 在"被测代码内部"拼换行，避免多层转义把人绕晕
  const t6 = run('JSON.stringify((function(){'
    + "var P = normalizeParity({ status:'fail', tol:1e-6, stdout:"
    + "'[fft] 最大相对误差 3.1e-15' + String.fromCharCode(10) + '[beamform] 最大相对误差 2.5e-5' });"
    + 'return { st:P.status, n:Object.keys(P.errors).length, fft:P.errors.fft, bf:P.errors.beamform }; })())');
  const o6 = JSON.parse(t6);
  ck(o6.n === 2 && o6.fft === 3.1e-15 && o6.bf > 1e-6 && o6.st === 'fail',
    'parity 从 stdout 兜底解析逐模块误差（并按 tol 判失败）：' + t6);

  console.log('LOG: ' + JSON.stringify(run('state.log'), null, 1).slice(0, 1500));
  console.log('\n[' + MODE + '] canvas putImageData 次数: ' + PUTS.length + '，错误日志: ' + run('state.log.length'));
  if (fails.length) { console.log('FAILED ' + fails.length + ' checks'); process.exit(1); }
  console.log('DOM SMOKE TEST PASSED (' + MODE + ')');
  process.exit(0);
})().catch(e => { console.error('TEST CRASH', e); process.exit(2); });
