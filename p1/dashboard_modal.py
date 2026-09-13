"""
Shared themed modal dialogs for IoC dashboards (P1–P3 + Bonus).
Replaces native window.alert / window.confirm with dark-theme overlays.
"""

# Insert into dashboard f-strings as {MODAL_CSS}, {MODAL_HTML}, {MODAL_JS}.
# Values use normal braces; Python only parses braces in the f-string literal.

MODAL_CSS = """
        /* Themed dialog (replaces native alert/confirm) */
        .ioc-modal-overlay {
            display: none; position: fixed; inset: 0; z-index: 9999;
            background: rgba(2, 6, 23, 0.72); backdrop-filter: blur(4px);
            align-items: center; justify-content: center; padding: 24px;
        }
        .ioc-modal-overlay.open { display: flex; }
        .ioc-modal {
            background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
            max-width: 440px; width: 100%; box-shadow: 0 20px 50px rgba(0,0,0,0.45);
            overflow: hidden; animation: iocModalIn 0.18s ease-out;
        }
        @keyframes iocModalIn {
            from { opacity: 0; transform: translateY(8px) scale(0.98); }
            to { opacity: 1; transform: none; }
        }
        .ioc-modal-header {
            display: flex; align-items: center; gap: 10px; padding: 14px 18px;
            border-bottom: 1px solid var(--border); background: rgba(255,255,255,0.02);
        }
        .ioc-modal-header .ioc-modal-icon {
            width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center;
            justify-content: center; font-size: 14px; font-weight: 700; flex-shrink: 0;
        }
        .ioc-modal.info .ioc-modal-icon {
            background: rgba(56, 189, 248, 0.15); color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        .ioc-modal.success .ioc-modal-icon {
            background: rgba(74, 222, 128, 0.15); color: var(--accent-green);
            border: 1px solid rgba(74, 222, 128, 0.35);
        }
        .ioc-modal.warn .ioc-modal-icon {
            background: rgba(251, 191, 36, 0.15); color: var(--accent-amber);
            border: 1px solid rgba(251, 191, 36, 0.35);
        }
        .ioc-modal.error .ioc-modal-icon {
            background: rgba(248, 113, 113, 0.15); color: var(--accent-red);
            border: 1px solid rgba(248, 113, 113, 0.35);
        }
        .ioc-modal-title { font-size: 15px; font-weight: 700; color: #fff; }
        .ioc-modal-body {
            padding: 16px 18px; font-size: 14px; color: var(--text-muted);
            white-space: pre-wrap; line-height: 1.55;
        }
        .ioc-modal-footer {
            display: flex; justify-content: flex-end; gap: 10px; padding: 12px 18px 16px;
            border-top: 1px solid var(--border);
        }
        /* Modal action buttons (P1 may not define .btn elsewhere) */
        .ioc-modal-footer .btn {
            padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: 600;
            cursor: pointer; border: none; display: inline-flex; align-items: center;
            gap: 8px; font-family: inherit;
        }
        .ioc-modal-footer .btn-primary {
            background: var(--accent, #38bdf8); color: #0f172a;
        }
        .ioc-modal-footer .btn-primary:hover { filter: brightness(1.08); }
        .ioc-modal-footer .btn-secondary {
            background: #334155; color: var(--text, #f8fafc);
        }
        .ioc-modal-footer .btn-secondary:hover { background: #475569; }
        .ioc-modal-footer .btn-danger {
            background: rgba(248, 113, 113, 0.2); color: var(--accent-red, #f87171);
            border: 1px solid rgba(248, 113, 113, 0.4);
        }
        .ioc-modal-footer .btn-danger:hover { background: rgba(248, 113, 113, 0.35); }
"""

MODAL_HTML = """
    <div id="ioc-modal-overlay" class="ioc-modal-overlay" role="dialog" aria-modal="true"
         aria-labelledby="ioc-modal-title">
        <div id="ioc-modal" class="ioc-modal info">
            <div class="ioc-modal-header">
                <div id="ioc-modal-icon" class="ioc-modal-icon">i</div>
                <div id="ioc-modal-title" class="ioc-modal-title">Notice</div>
            </div>
            <div id="ioc-modal-body" class="ioc-modal-body"></div>
            <div class="ioc-modal-footer" id="ioc-modal-footer"></div>
        </div>
    </div>
"""

MODAL_JS = """
        const IOC_MODAL = {
            overlay: null, resolve: null,
            icons: { info: 'i', success: '\\u2713', warn: '!', error: '\\u2715' },
            titles: { info: 'Notice', success: 'Success', warn: 'Confirm', error: 'Error' }
        };

        function _iocEnsureModal() {
            if (!IOC_MODAL.overlay) IOC_MODAL.overlay = document.getElementById('ioc-modal-overlay');
            return IOC_MODAL.overlay;
        }

        function _iocCloseModal(result) {
            const overlay = _iocEnsureModal();
            if (overlay) overlay.classList.remove('open');
            const r = IOC_MODAL.resolve;
            IOC_MODAL.resolve = null;
            if (r) r(result);
        }

        function showDialog(message, options) {
            options = options || {};
            const kind = options.kind || 'info';
            const title = options.title || IOC_MODAL.titles[kind] || 'Notice';
            const confirmLabel = options.confirmLabel || 'OK';
            const cancelLabel = options.cancelLabel || 'Cancel';
            const showCancel = !!options.showCancel;

            return new Promise(function(resolve) {
                IOC_MODAL.resolve = resolve;
                const overlay = _iocEnsureModal();
                const modal = document.getElementById('ioc-modal');
                const icon = document.getElementById('ioc-modal-icon');
                const titleEl = document.getElementById('ioc-modal-title');
                const body = document.getElementById('ioc-modal-body');
                const footer = document.getElementById('ioc-modal-footer');

                modal.className = 'ioc-modal ' + kind;
                icon.textContent = IOC_MODAL.icons[kind] || 'i';
                titleEl.textContent = title;
                body.textContent = message;

                footer.innerHTML = '';
                if (showCancel) {
                    const cancelBtn = document.createElement('button');
                    cancelBtn.className = 'btn btn-secondary';
                    cancelBtn.textContent = cancelLabel;
                    cancelBtn.onclick = function() { _iocCloseModal(false); };
                    footer.appendChild(cancelBtn);
                }
                const okBtn = document.createElement('button');
                okBtn.className = kind === 'error' ? 'btn btn-danger' : 'btn btn-primary';
                okBtn.textContent = confirmLabel;
                okBtn.onclick = function() { _iocCloseModal(true); };
                footer.appendChild(okBtn);

                overlay.classList.add('open');
                okBtn.focus();
            });
        }

        function showAlert(message, kind, title) {
            return showDialog(message, { kind: kind || 'info', title: title, showCancel: false });
        }

        function showConfirm(message, title) {
            return showDialog(message, {
                kind: 'warn', title: title || 'Confirm', showCancel: true,
                confirmLabel: 'Confirm', cancelLabel: 'Cancel'
            });
        }

        (function() {
            var overlay = document.getElementById('ioc-modal-overlay');
            if (overlay) {
                overlay.addEventListener('click', function(e) {
                    if (e.target === e.currentTarget) _iocCloseModal(false);
                });
            }
            document.addEventListener('keydown', function(e) {
                var o = _iocEnsureModal();
                if (e.key === 'Escape' && o && o.classList.contains('open')) {
                    _iocCloseModal(false);
                }
            });
        })();
"""
