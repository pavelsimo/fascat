#!/usr/bin/env node
/**
 * docs site builder for fascat
 *
 * Pure Node.js — zero external dependencies.
 * Reads docs/*.md → outputs a polished static site to dist/docs-site/
 *
 * Features: sidebar nav (with sections), sticky ToC, dark/light toggle,
 * syntax highlighting, copy buttons, hero on index page, color themes,
 * llms.txt, .nojekyll, CNAME support.
 */

import {
  readFileSync, writeFileSync, mkdirSync, existsSync,
} from "fs";
import { join, basename } from "path";

// ── Config ────────────────────────────────────────────────────────────────────

const TOOL         = "fascat";
const REPO_URL     = "https://github.com/pavelsimo/fascat";
const BREW_TAP     = "pavelsimo/homebrew-tap";
const DESC         = "convert CAD STEP data into realtime-ready OpenUSD and glTF assets";
const COLOR_SCHEME = "teal"; // teal | ocean | purple | amber
const SITE_BASE    = existsSync("docs/CNAME")
  ? `https://${readFileSync("docs/CNAME","utf8").trim()}`
  : `https://pavelsimo.github.io/fascat`;

const SRC = "docs";
const OUT = join("dist", "docs-site");
mkdirSync(OUT, { recursive: true });

// ── Color themes ──────────────────────────────────────────────────────────────

const THEMES = {
  teal: {
    dark:  { accent:"#2dd4bf", accentSoft:"rgba(45,212,191,.16)", accentStrong:"#5eead4",
             bg:"#0a0e16", paper:"#141a26", surface2:"#1a212d", line:"#232a38", lineSoft:"#1a212d",
             ink:"#f1f5f9", text:"#cbd2dc", muted:"#8d96a4", codeBg:"#04080f" },
    light: { accent:"#0f766e", accentSoft:"rgba(15,118,110,.10)", accentStrong:"#0d5b55",
             bg:"#fafbfc", paper:"#ffffff", surface2:"#eef1f5", line:"#e3e7ec", lineSoft:"#eef1f5",
             ink:"#0f172a", text:"#1f2937", muted:"#5b6470", codeBg:"#0a1322" },
  },
  ocean: {
    dark:  { accent:"#60a5fa", accentSoft:"rgba(96,165,250,.16)", accentStrong:"#93c5fd",
             bg:"#0c1220", paper:"#111827", surface2:"#1e293b", line:"#1e293b", lineSoft:"#172033",
             ink:"#f1f5f9", text:"#cbd5e1", muted:"#64748b", codeBg:"#060c18" },
    light: { accent:"#2563eb", accentSoft:"rgba(37,99,235,.10)", accentStrong:"#1d4ed8",
             bg:"#f8fafc", paper:"#ffffff", surface2:"#eff6ff", line:"#e2e8f0", lineSoft:"#eff6ff",
             ink:"#0f172a", text:"#1e293b", muted:"#64748b", codeBg:"#0a1220" },
  },
  purple: {
    dark:  { accent:"#a78bfa", accentSoft:"rgba(167,139,250,.16)", accentStrong:"#c4b5fd",
             bg:"#0d0b1a", paper:"#160f28", surface2:"#1e1535", line:"#2a1f45", lineSoft:"#1e1535",
             ink:"#f5f3ff", text:"#ddd6fe", muted:"#8b7ec8", codeBg:"#07050f" },
    light: { accent:"#7c3aed", accentSoft:"rgba(124,58,237,.10)", accentStrong:"#6d28d9",
             bg:"#faf8ff", paper:"#ffffff", surface2:"#f3f0ff", line:"#e9e3ff", lineSoft:"#f3f0ff",
             ink:"#1e0a3c", text:"#2d1b69", muted:"#6b7280", codeBg:"#0d0b1a" },
  },
  amber: {
    dark:  { accent:"#fbbf24", accentSoft:"rgba(251,191,36,.14)", accentStrong:"#fcd34d",
             bg:"#0f0c05", paper:"#1c1608", surface2:"#2a1f08", line:"#3a2c0a", lineSoft:"#2a1f08",
             ink:"#fffbeb", text:"#fde68a", muted:"#a16207", codeBg:"#080600" },
    light: { accent:"#d97706", accentSoft:"rgba(217,119,6,.10)", accentStrong:"#b45309",
             bg:"#fffbf0", paper:"#ffffff", surface2:"#fef3c7", line:"#fde68a", lineSoft:"#fef3c7",
             ink:"#451a03", text:"#92400e", muted:"#78350f", codeBg:"#0f0c05" },
  },
};

const theme = THEMES[COLOR_SCHEME] || THEMES.teal;

function themeCss(t) {
  const d = t.dark, l = t.light;
  return `[data-theme=dark]{
  --bg:${d.bg};--paper:${d.paper};--surface2:${d.surface2};
  --line:${d.line};--line-soft:${d.lineSoft};
  --ink:${d.ink};--text:${d.text};--muted:${d.muted};
  --accent:${d.accent};--accent-soft:${d.accentSoft};--accent-strong:${d.accentStrong};
  --code-bg:${d.codeBg};--code-fg:#e6edf3;--code-border:#1f2937;
}
[data-theme=light]{
  --bg:${l.bg};--paper:${l.paper};--surface2:${l.surface2};
  --line:${l.line};--line-soft:${l.lineSoft};
  --ink:${l.ink};--text:${l.text};--muted:${l.muted};
  --accent:${l.accent};--accent-soft:${l.accentSoft};--accent-strong:${l.accentStrong};
  --code-bg:${l.codeBg};--code-fg:#e6edf3;--code-border:#1f2937;
}`;
}

// ── Markdown parser ───────────────────────────────────────────────────────────

function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function stash(s, buf) { const k=`\x00${buf.length}\x00`; buf.push(s); return k; }
function unstash(s, buf) { return s.replace(/\x00(\d+)\x00/g,(_,i)=>buf[+i]); }

// ── Syntax highlighting ───────────────────────────────────────────────────────

function hlBash(code) {
  const b=[]; let s=esc(code);
  s=s.replace(/"(?:[^"\\]|\\.)*"/g, m=>stash(`<span class=hs>${m}</span>`,b));
  s=s.replace(/'[^']*'/g,           m=>stash(`<span class=hs>${m}</span>`,b));
  s=s.replace(/(#.*)$/gm,           m=>stash(`<span class=hc>${m}</span>`,b));
  s=s.replace(/(--?[\w-]+=?\S*)/g,  m=>stash(`<span class=hf>${m}</span>`,b));
  s=s.replace(/^(\s*)(\$\s+)?(\w[\w.-]*)/gm,
    (_,sp,pr,cmd)=>`${sp}${pr||""}<span class=hk>${cmd}</span>`);
  return unstash(s,b);
}

function hlGo(code) {
  const kw=/\b(package|import|func|type|struct|interface|var|const|return|if|else|for|range|switch|case|default|break|continue|go|defer|select|chan|map|make|new|nil|true|false|error|string|int|bool|byte|rune|any)\b/g;
  const b=[]; let s=esc(code);
  s=s.replace(/("(?:[^"\\]|\\.)*"|`[^`]*`)/g, m=>stash(`<span class=hs>${m}</span>`,b));
  s=s.replace(/(\/\/.*)$/gm,                   m=>stash(`<span class=hc>${m}</span>`,b));
  s=s.replace(kw,                               m=>`<span class=hk>${m}</span>`);
  s=s.replace(/\b(\d+)\b/g,                     `<span class=hn>$1</span>`);
  return unstash(s,b);
}

function hlYaml(code) {
  const b=[]; let s=esc(code);
  s=s.replace(/(#.*)$/gm,   m=>stash(`<span class=hc>${m}</span>`,b));
  s=s.replace(/^(\s*)([\w-]+)(\s*:)/gm,
    (_,sp,k,col)=>`${sp}<span class=hk>${k}</span>${col}`);
  s=s.replace(/:\s*(.+)$/gm,
    (m,v)=>m.replace(v,`<span class=hs>${v}</span>`));
  return unstash(s,b);
}

function hlJson(code) {
  const b=[]; let s=esc(code);
  s=s.replace(/"(?:[^"\\]|\\.)*"/g, m=>stash(`<span class=hs>${m}</span>`,b));
  s=s.replace(/\b(true|false|null)\b/g, `<span class=hk>$1</span>`);
  s=s.replace(/\b(\d+\.?\d*)\b/g,       `<span class=hn>$1</span>`);
  return unstash(s,b);
}

function highlight(lang, code) {
  if (lang==="bash"||lang==="sh"||lang==="zsh") return hlBash(code);
  if (lang==="go")   return hlGo(code);
  if (lang==="yaml"||lang==="yml") return hlYaml(code);
  if (lang==="json") return hlJson(code);
  return esc(code);
}

// ── Inline Markdown ───────────────────────────────────────────────────────────

function inline(text, buf) {
  text=text.replace(/`([^`]+)`/g,
    (_,c)=>stash(`<code>${esc(c)}</code>`,buf));
  text=text.replace(/\*\*\*(.+?)\*\*\*/g,"<strong><em>$1</em></strong>");
  text=text.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>");
  text=text.replace(/\*(.+?)\*/g,"<em>$1</em>");
  text=text.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(_,label,href)=>{
    const ext=href.startsWith("http")?` target="_blank" rel="noopener"`:"";
    return stash(`<a href="${href}"${ext}>${label}</a>`,buf);
  });
  return text;
}

// ── Markdown → HTML ───────────────────────────────────────────────────────────

function slugify(t) {
  return t.toLowerCase().replace(/[^\w\s-]/g,"").trim().replace(/\s+/g,"-");
}

function parse(src) {
  const lines = src.split("\n");
  const buf=[], toc=[], out=[];
  let i=0, inPara=false, inUl=false, inOl=false, inFence=false;
  let fLang="", fLines=[];

  if (lines[0]==="---") {
    i=1; while(i<lines.length&&lines[i]!=="---") i++; i++;
  }

  function flushPara()  { if(inPara){out.push("</p>");inPara=false;} }
  function flushUl()    { if(inUl){out.push("</ul>");inUl=false;} }
  function flushOl()    { if(inOl){out.push("</ol>");inOl=false;} }
  function flushBlock() { flushPara();flushUl();flushOl(); }

  for(;i<lines.length;i++) {
    const line=lines[i];

    if(line.startsWith("```")) {
      if(!inFence) {
        flushBlock();
        inFence=true; fLang=line.slice(3).trim(); fLines=[];
      } else {
        const body=highlight(fLang,fLines.join("\n"));
        out.push(`<div class="code-wrap"><pre><code>${body}</code></pre>`+
          `<button class="copy-btn">Copy</button></div>`);
        inFence=false; fLines=[];
      }
      continue;
    }
    if(inFence){fLines.push(line);continue;}

    const hm=line.match(/^(#{1,4})\s+(.*)/);
    if(hm) {
      flushBlock();
      const lvl=hm[1].length, rawText=hm[2];
      const id=slugify(rawText);
      const text=unstash(inline(rawText,buf),buf);
      if(lvl<=3) toc.push({level:lvl,id,text:rawText});
      out.push(`<h${lvl} id="${id}"><a class="anchor" href="#${id}">#</a>${text}</h${lvl}>`);
      continue;
    }

    if(line.startsWith(">")) {
      flushBlock();
      const text=unstash(inline(line.slice(1).trim(),buf),buf);
      out.push(`<blockquote><p>${text}</p></blockquote>`);
      continue;
    }

    if(line.match(/^-{3,}$/)){flushBlock();out.push("<hr>");continue;}

    if(line.startsWith("|")) {
      flushBlock();
      const rows=[line];
      while(i+1<lines.length&&lines[i+1].startsWith("|")) rows.push(lines[++i]);
      out.push('<table>');
      rows.forEach((row,ri)=>{
        if(row.match(/^\|[-| :]+\|$/)) return;
        const cells=row.split("|").slice(1,-1);
        const tag=ri===0?"th":"td";
        out.push("<tr>"+cells.map(c=>`<${tag}>${unstash(inline(c.trim(),buf),buf)}</${tag}>`).join("")+"</tr>");
      });
      out.push("</table>");
      continue;
    }

    if(line.match(/^[-*]\s/)) {
      flushPara(); flushOl();
      if(!inUl){out.push("<ul>");inUl=true;}
      out.push(`<li>${unstash(inline(line.replace(/^[-*]\s/,""),buf),buf)}</li>`);
      continue;
    }

    if(line.match(/^\d+\.\s/)) {
      flushPara(); flushUl();
      if(!inOl){out.push("<ol>");inOl=true;}
      out.push(`<li>${unstash(inline(line.replace(/^\d+\.\s/,""),buf),buf)}</li>`);
      continue;
    }

    if(line.trim()===""){flushBlock();continue;}

    flushUl();flushOl();
    if(!inPara){out.push("<p>");inPara=true;} else out.push(" ");
    out.push(unstash(inline(line,buf),buf));
  }
  flushBlock();
  return {html:out.join("\n"),toc};
}

// ── Page structure ────────────────────────────────────────────────────────────

function tocHtml(toc) {
  if(toc.length<2) return "";
  const items=toc.map(({id,text,level})=>
    `<li class="toc-${level}"><a href="#${id}">${esc(text)}</a></li>`
  ).join("\n");
  return `<nav class="toc" aria-label="On this page">
  <p class="toc-title">On this page</p>
  <ul>${items}</ul>
</nav>`;
}

function sidebarHtml(pages, currentSlug) {
  const slugSection = {};
  for (const [sectionLabel, files] of sections) {
    for (const f of files) slugSection[basename(f, ".md")] = sectionLabel;
  }

  let items = "", lastSection = null;
  for (const { slug, label } of pages) {
    const sec = slugSection[slug];
    if (sec !== lastSection) {
      items += `<li class="nav-group"><h2>${esc(sec)}</h2></li>\n`;
      lastSection = sec;
    }
    const active = slug === currentSlug ? ' class="active"' : "";
    items += `<li><a href="${slug}.html"${active}>${esc(label)}</a></li>\n`;
  }

  return `<nav class="sidebar" id="sidebar" aria-label="Site navigation">
  <div class="sidebar-brand">
    <a href="index.html" class="brand-link">
      <span class="brand-mark">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13zM0 8a8 8 0 1116 0A8 8 0 010 8z"/><path d="M8 6.5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 018 6.5zM8 5a1 1 0 100-2 1 1 0 000 2z"/></svg>
      </span>
      ${TOOL}
    </a>
  </div>
  <ul class="sidebar-nav">${items}</ul>
  <div class="sidebar-footer">
    <a href="${REPO_URL}" target="_blank" rel="noopener" class="gh-link">
      <svg height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
      GitHub
    </a>
    <button class="theme-btn" id="themeBtn" aria-label="Toggle theme">
      <span class="icon-sun">☀️</span>
      <span class="icon-moon">🌙</span>
      <span id="themeLabel">Light</span>
    </button>
  </div>
</nav>`;
}

function heroHtml() {
  return `<section class="hero">
  <p class="hero-eyebrow">CLI Tool</p>
  <h1 class="hero-title">${TOOL}</h1>
  <p class="hero-desc">${esc(DESC)}</p>
  <div class="hero-actions">
    <a class="btn-primary" href="install.html">Get started</a>
    <a class="btn-outline" href="${REPO_URL}" target="_blank" rel="noopener">GitHub</a>
  </div>
  <div class="install-cmd">
    <span class="install-label">Install</span>
    <code>brew install ${TOOL}</code>
    <button class="copy-btn" data-copy="brew install ${TOOL}">Copy</button>
  </div>
</section>`;
}

// ── Full page HTML ────────────────────────────────────────────────────────────

function renderPage({slug, title, bodyHtml, toc, pages, isIndex}) {
  const sidebar  = sidebarHtml(pages, slug);
  const tocBlock = tocHtml(toc);
  const hero     = isIndex ? heroHtml() : "";
  const pageTitle= isIndex ? TOOL : `${title} — ${TOOL}`;

  return `<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${pageTitle}</title>
<meta name="description" content="${esc(DESC)}">
<meta property="og:type" content="website">
<meta property="og:title" content="${pageTitle}">
<meta property="og:description" content="${esc(DESC)}">
<meta property="og:url" content="${SITE_BASE}/${slug === "index" ? "" : slug + ".html"}">
<meta name="twitter:card" content="summary">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── Reset + tokens ─────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --sidebar-w:280px;
  --font-sans:"Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"JetBrains Mono","SF Mono",ui-monospace,monospace;
}
${themeCss(theme)}

/* ── Base ───────────────────────────────────────────── */
html{scroll-behavior:smooth;font-size:16px}
body{
  background:var(--bg);color:var(--text);
  font:1rem/1.65 var(--font-sans);min-height:100vh;
  transition:background-color .18s,color .18s;
}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
img{max-width:100%}
hr{border:none;border-top:1px solid var(--line);margin:2rem 0}

/* ── Shell grid ─────────────────────────────────────── */
.shell{
  display:grid;
  grid-template-columns:var(--sidebar-w) minmax(0,1fr);
  min-height:100vh;
}

/* ── Sidebar ────────────────────────────────────────── */
.sidebar{
  position:sticky;top:0;height:100vh;overflow-y:auto;
  background:var(--paper);border-right:1px solid var(--line);
  padding:24px 22px;
  display:flex;flex-direction:column;
  scrollbar-width:thin;scrollbar-color:var(--line) transparent;
}
.sidebar::-webkit-scrollbar{width:6px}
.sidebar::-webkit-scrollbar-thumb{background:var(--line);border-radius:6px}

/* brand */
.sidebar-brand{margin-bottom:1.25rem}
.brand-link{
  display:flex;align-items:center;gap:10px;
  font-weight:700;font-size:1.05rem;color:var(--ink);
}
.brand-link:hover{color:var(--accent);text-decoration:none}
.brand-mark{
  width:30px;height:30px;border-radius:8px;
  background:var(--accent);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.brand-mark svg{fill:#fff}
[data-theme=dark] .brand-mark svg{fill:#0a0e16}

/* sidebar nav */
.sidebar-nav{list-style:none;flex:1}
.nav-group h2{
  font-size:.68rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin:1.25rem 0 .3rem;padding:0 10px;
}
.nav-group:first-child h2{margin-top:.25rem}
.sidebar-nav a{
  display:block;padding:5px 10px;margin:1px 0;
  border-radius:6px;font-size:.9rem;line-height:1.4;
  color:var(--text);transition:background .12s,color .12s;
}
.sidebar-nav a:hover{background:var(--line-soft);color:var(--ink);text-decoration:none}
.sidebar-nav a.active{background:var(--accent-soft);color:var(--accent);font-weight:600}

/* sidebar footer */
.sidebar-footer{
  margin-top:auto;padding-top:1rem;
  border-top:1px solid var(--line);
  display:flex;align-items:center;gap:.75rem;
}
.gh-link{
  display:flex;align-items:center;gap:.4rem;
  font-size:.8rem;color:var(--muted);flex:1;
}
.gh-link:hover{color:var(--text);text-decoration:none}
.theme-btn{
  background:none;border:1px solid var(--line);border-radius:8px;
  cursor:pointer;padding:.3rem .5rem;color:var(--muted);font-size:.75rem;
  display:flex;align-items:center;gap:.3rem;white-space:nowrap;
  font-family:var(--font-sans);
  transition:border-color .15s,color .15s;
}
.theme-btn:hover{color:var(--ink);border-color:var(--accent)}
.icon-sun,.icon-moon{font-size:.85rem}
[data-theme=dark]  .icon-moon{display:inline}[data-theme=dark]  .icon-sun{display:none}
[data-theme=light] .icon-sun{display:inline}[data-theme=light] .icon-moon{display:none}

/* ── Body column ────────────────────────────────────── */
.body-col{display:flex;flex-direction:column;min-width:0}

/* mobile bar — hidden on desktop */
.mob-bar{
  display:none;
  position:sticky;top:0;z-index:100;
  background:var(--paper);border-bottom:1px solid var(--line);
  padding:.6rem 1rem;align-items:center;gap:.75rem;height:52px;
}
.mob-brand{font-weight:700;font-size:.95rem;color:var(--ink);flex:1}
.mob-brand:hover{color:var(--accent);text-decoration:none}
.hamburger{
  background:none;border:none;cursor:pointer;
  color:var(--muted);padding:.25rem;display:flex;
}

/* ── Main + ToC row ─────────────────────────────────── */
.content-row{display:flex;flex:1;min-width:0}
.main{
  flex:1;min-width:0;
  padding:32px clamp(20px,4.5vw,56px) 80px;
}

/* ── ToC ────────────────────────────────────────────── */
.toc{
  width:210px;flex-shrink:0;
  position:sticky;top:0;height:100vh;overflow-y:auto;
  padding:32px 24px 32px 14px;
  border-left:1px solid var(--line);
  font-size:.84rem;
}
.toc-title{
  font-size:.68rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted);margin-bottom:.75rem;
}
.toc ul{list-style:none}
.toc a{
  display:block;padding:4px 0 4px 10px;
  border-left:2px solid transparent;margin-left:-12px;
  color:var(--muted);transition:color .12s,border-color .12s;
}
.toc a:hover,.toc a.active{
  color:var(--accent);border-left-color:var(--accent);text-decoration:none;
}
.toc-3 a{padding-left:22px;font-size:.8rem}

/* ── Hero ───────────────────────────────────────────── */
.hero{
  padding:14px 0 28px;
  border-bottom:1px solid var(--line);
  margin-bottom:2rem;
}
.hero-eyebrow{
  font-size:.7rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.08em;color:var(--accent);margin-bottom:.85rem;
}
.hero-title{
  font-size:clamp(2rem,4vw,3.1rem);font-weight:700;
  letter-spacing:-.01em;line-height:1.04;
  color:var(--ink);margin-bottom:.45rem;
}
.hero-desc{
  font-size:1.1rem;color:var(--text);
  max-width:62ch;margin-bottom:1.5rem;line-height:1.55;
}
.hero-actions{display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1.5rem}
.btn-primary{
  display:inline-flex;align-items:center;gap:7px;
  background:var(--accent);border:1px solid var(--accent);
  color:#fff;font-weight:600;padding:10px 18px;
  border-radius:8px;font-size:.92rem;
  transition:opacity .15s;
}
[data-theme=dark] .btn-primary{color:#0a0e16}
.btn-primary:hover{opacity:.85;text-decoration:none}
.btn-outline{
  display:inline-flex;align-items:center;gap:7px;
  border:1px solid var(--line);color:var(--text);
  padding:10px 18px;border-radius:8px;font-size:.92rem;font-weight:500;
  transition:border-color .15s,color .15s;
}
.btn-outline:hover{border-color:var(--accent);color:var(--ink);text-decoration:none}
.install-cmd{
  display:inline-flex;align-items:center;gap:12px;
  background:var(--code-bg);color:var(--code-fg);
  border:1px solid var(--code-border);
  padding:10px 10px 10px 16px;border-radius:8px;
  font:500 .9rem/1.2 var(--font-mono);max-width:32em;
}
.install-cmd code{background:none;border:none;padding:0;color:inherit;font-size:inherit}
.install-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}

/* ── Typography ─────────────────────────────────────── */
.main h1{font-size:2.5rem;font-weight:700;line-height:1.08;color:var(--ink);margin-bottom:1rem;letter-spacing:-.02em}
.main h2{font-size:1.4rem;font-weight:600;line-height:1.2;color:var(--ink);border-bottom:1px solid var(--line);padding-bottom:.4rem;margin:2.5rem 0 1rem;scroll-margin-top:24px}
.main h3{font-size:1.08rem;font-weight:600;color:var(--ink);margin:2rem 0 .6rem;scroll-margin-top:24px}
.main h4{font-size:.96rem;font-weight:600;color:var(--muted);margin:1.5rem 0 .5rem;scroll-margin-top:24px}
.main p{margin-bottom:1.05em}
.main ul,.main ol{padding-left:1.4rem;margin-bottom:1em}
.main li{margin-bottom:.3rem}
.main blockquote{
  border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:10px 16px;border-radius:0 8px 8px 0;margin:1.4em 0;color:var(--text);
}
.anchor{opacity:0;font-size:.8em;color:var(--muted);margin-right:.4rem;text-decoration:none}
h2:hover .anchor,h3:hover .anchor{opacity:1}

/* ── Inline code ────────────────────────────────────── */
code{
  background:var(--surface2);border-radius:5px;
  padding:.15em .4em;font-family:var(--font-mono);font-size:.84em;
}

/* ── Code blocks ────────────────────────────────────── */
.code-wrap{position:relative;margin:1.25rem 0}
pre{
  background:var(--code-bg);border:1px solid var(--code-border);
  border-radius:8px;padding:14px 18px;
  overflow-x:auto;font-family:var(--font-mono);font-size:.85rem;line-height:1.6;
  color:var(--code-fg);
  scrollbar-width:thin;scrollbar-color:#334155 transparent;
}
pre code{background:none;border:none;padding:0;font-size:inherit;color:inherit}
.copy-btn{
  position:absolute;top:.5rem;right:.5rem;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);
  color:#8d96a4;font-size:.72rem;padding:5px 11px;border-radius:6px;
  cursor:pointer;font:500 .72rem/1 var(--font-sans);
  transition:background .12s,color .12s;
}
.copy-btn:hover{background:rgba(255,255,255,.16);color:#e6edf3}
.copy-btn.ok{background:var(--accent);border-color:var(--accent);color:#0a0e16}

/* ── Syntax highlight tokens ────────────────────────── */
.hk{color:#e387cb}
.hs{color:#a8e0a3}
.hn{color:#f6c177}
.hc{color:#7c8597;font-style:italic}
.hf{color:#fcd28a}

/* ── Tables ─────────────────────────────────────────── */
table{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.9rem}
th{
  background:var(--line-soft);text-align:left;
  padding:9px 10px;border-bottom:1px solid var(--line);
  font-size:.8rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.04em;color:var(--muted);
}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
td code{font-size:.8em}

/* ── Responsive ─────────────────────────────────────── */
@media(max-width:1200px){.toc{display:none}}
@media(max-width:900px){
  .shell{grid-template-columns:1fr}
  .sidebar{
    position:fixed;top:0;left:0;z-index:200;
    height:100vh;width:var(--sidebar-w);
    transform:translateX(-100%);transition:transform .22s ease;
  }
  .sidebar.open{transform:translateX(0)}
  .mob-bar{display:flex}
  .main{padding:20px 18px 56px}
}
</style>
</head>
<body>
<div class="shell">
  ${sidebar}
  <div class="body-col">
    <div class="mob-bar">
      <button class="hamburger" id="ham" aria-label="Toggle menu">
        <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="3" y1="6"  x2="17" y2="6"/>
          <line x1="3" y1="10" x2="17" y2="10"/>
          <line x1="3" y1="14" x2="17" y2="14"/>
        </svg>
      </button>
      <a class="mob-brand" href="index.html">${TOOL}</a>
    </div>
    <div class="content-row">
      <main class="main">
        ${hero}
        ${bodyHtml}
      </main>
      ${tocBlock}
    </div>
  </div>
</div>

<script>
// theme toggle
const root=document.documentElement;
const btn=document.getElementById("themeBtn");
const label=document.getElementById("themeLabel");
const stored=localStorage.getItem("theme")||"dark";
root.dataset.theme=stored;
label.textContent=stored==="dark"?"Light":"Dark";
btn.addEventListener("click",()=>{
  const next=root.dataset.theme==="dark"?"light":"dark";
  root.dataset.theme=next;
  localStorage.setItem("theme",next);
  label.textContent=next==="dark"?"Light":"Dark";
});

// hamburger
const ham=document.getElementById("ham");
const sidebar=document.getElementById("sidebar");
ham.addEventListener("click",()=>sidebar.classList.toggle("open"));
document.addEventListener("click",e=>{
  if(!sidebar.contains(e.target)&&!ham.contains(e.target))
    sidebar.classList.remove("open");
});

// copy buttons
document.querySelectorAll(".copy-btn").forEach(btn=>{
  btn.addEventListener("click",()=>{
    const pre=btn.previousElementSibling;
    const text=btn.dataset.copy||(pre&&pre.innerText)||"";
    navigator.clipboard.writeText(text.trim()).then(()=>{
      btn.textContent="Copied!";btn.classList.add("ok");
      setTimeout(()=>{btn.textContent="Copy";btn.classList.remove("ok");},2000);
    });
  });
});

// ToC scroll-spy
const tocLinks=[...document.querySelectorAll(".toc a")];
if(tocLinks.length){
  const heads=[...document.querySelectorAll("h2[id],h3[id]")];
  const obs=new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        tocLinks.forEach(a=>a.classList.remove("active"));
        const a=document.querySelector('.toc a[href="#'+e.target.id+'"]');
        if(a)a.classList.add("active");
      }
    });
  },{rootMargin:"0px 0px -70% 0px"});
  heads.forEach(h=>obs.observe(h));
}
</script>
</body>
</html>`;
}

// ── Navigation ────────────────────────────────────────────────────────────────

const sections = [
  ["Get Started", ["index.md", "install.md", "quickstart.md", "api.md"]],
  ["Reference",   ["reference.md"]],
];

const LABELS = {
  "index":      "Home",
  "quickstart": "Quick Start",
  "api":        "Python API",
};

function fileToLabel(filename) {
  const slug = basename(filename, ".md");
  if (LABELS[slug]) return LABELS[slug];
  return slug
    .replace(/-/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

const pages = sections.flatMap(([, files]) =>
  files.map(f => ({ slug: basename(f, ".md"), label: fileToLabel(f), file: f }))
);

// ── Build ─────────────────────────────────────────────────────────────────────

for(const {slug,label,file} of pages) {
  const filePath = join(SRC, file);
  if (!existsSync(filePath)) {
    console.error(`  ERROR: docs/${file} not found — check sections array`);
    process.exit(1);
  }
  const src    = readFileSync(filePath,"utf8");
  const {html,toc} = parse(src);
  const output = renderPage({
    slug, title:label, bodyHtml:html, toc, pages,
    isIndex: slug==="index",
  });
  writeFileSync(join(OUT,`${slug}.html`), output);
  console.log(`  wrote ${slug}.html  (${toc.length} ToC entries)`);
}

// .nojekyll
writeFileSync(join(OUT,".nojekyll"),"");

// CNAME
if(existsSync(join(SRC,"CNAME")))
  writeFileSync(join(OUT,"CNAME"),readFileSync(join(SRC,"CNAME")));

// llms.txt
writeFileSync(join(OUT,"llms.txt"),
`# ${TOOL}

> ${DESC}

## Install

\`\`\`bash
brew tap ${BREW_TAP}
brew install ${TOOL}
\`\`\`

## Source

${REPO_URL}

## Docs

${SITE_BASE}/
`);

console.log(`\nSite built → ${OUT}/  (${pages.length} page${pages.length===1?"":"s"})`);
