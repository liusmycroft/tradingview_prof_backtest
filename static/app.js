const API = '/api';

async function api(path, opts = {}) {
    const res = await fetch(`${API}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Request failed');
    }
    return res.json();
}

function $(sel, ctx = document) { return ctx.querySelector(sel); }
function $$(sel, ctx = document) { return ctx.querySelectorAll(sel); }

/** Escape HTML to prevent XSS when using innerHTML. */
function esc(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
}

function toast(msg, duration = 3000) {
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), duration);
}

function formatDate(iso) {
    return new Date(iso).toLocaleString();
}

function getBacktestId() {
    const parts = window.location.pathname.split('/');
    const id = parts[2];
    if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error('Invalid backtest ID');
    return id;
}

function webhookUrl(id) {
    return `${window.location.origin}/webhook/${esc(id)}`;
}
