// GAP Compare Slider - before/after split-slider comparer
// by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
//
// Drawn straight on the LiteGraph node canvas (no DOM widgets), so it has
// no widget-serialization side effects and works at any zoom level.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PAD = 10;
const TOP = 54; // below the two input slots

function imgUrl(info) {
    return api.apiURL(
        `/view?filename=${encodeURIComponent(info.filename)}` +
        `&type=${info.type}&subfolder=${encodeURIComponent(info.subfolder || "")}` +
        `&rand=${Math.random()}`);
}

app.registerExtension({
    name: "geekatplay.genupscale.compare_slider",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GAPCompareSlider") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const load = (arr) => {
                if (!arr || !arr.length) return null;
                const img = new Image();
                img.src = imgUrl(arr[0]);
                img.onload = () => this.setDirtyCanvas(true, false);
                return img;
            };
            this._gapA = load(message.a_images);
            this._gapB = load(message.b_images);
            if (this._gapSplit === undefined) this._gapSplit = 0.5;
            this.setDirtyCanvas(true, false);
        };

        nodeType.prototype._gapRect = function () {
            const w = this.size[0] - PAD * 2;
            const h = this.size[1] - TOP - PAD;
            if (w < 20 || h < 20) return null;
            const img = this._gapA || this._gapB;
            if (!img || !img.naturalWidth) return null;
            const scale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
            const dw = img.naturalWidth * scale, dh = img.naturalHeight * scale;
            return {
                x: PAD + (w - dw) / 2,
                y: TOP + (h - dh) / 2,
                w: dw, h: dh,
            };
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags.collapsed) return;
            const r = this._gapRect();
            if (!r) {
                ctx.fillStyle = "#888";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Run the workflow to compare images",
                    this.size[0] / 2, TOP + 30);
                return;
            }
            const split = this._gapSplit ?? 0.5;
            const sx = r.x + r.w * split;

            // AFTER (b) fills the frame, BEFORE (a) is clipped to the left of the divider
            if (this._gapB?.naturalWidth) ctx.drawImage(this._gapB, r.x, r.y, r.w, r.h);
            if (this._gapA?.naturalWidth) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(r.x, r.y, r.w * split, r.h);
                ctx.clip();
                ctx.drawImage(this._gapA, r.x, r.y, r.w, r.h);
                ctx.restore();
            }

            // divider line
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(sx, r.y);
            ctx.lineTo(sx, r.y + r.h);
            ctx.stroke();

            // grab handle with arrows
            const cy = r.y + r.h / 2;
            ctx.fillStyle = "rgba(255,255,255,0.9)";
            ctx.beginPath();
            ctx.arc(sx, cy, 11, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = "#222";
            ctx.beginPath();
            ctx.moveTo(sx - 7, cy); ctx.lineTo(sx - 2, cy - 4); ctx.lineTo(sx - 2, cy + 4);
            ctx.moveTo(sx + 7, cy); ctx.lineTo(sx + 2, cy - 4); ctx.lineTo(sx + 2, cy + 4);
            ctx.fill();

            // labels
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "left";
            ctx.fillStyle = "rgba(0,0,0,0.55)";
            ctx.fillRect(r.x + 4, r.y + 4, 52, 16);
            ctx.fillStyle = "#fff";
            ctx.fillText("BEFORE", r.x + 8, r.y + 16);
            ctx.textAlign = "right";
            ctx.fillStyle = "rgba(0,0,0,0.55)";
            ctx.fillRect(r.x + r.w - 48, r.y + 4, 44, 16);
            ctx.fillStyle = "#fff";
            ctx.fillText("AFTER", r.x + r.w - 8, r.y + 16);
        };

        nodeType.prototype._gapSetSplit = function (pos) {
            const r = this._gapRect();
            if (!r) return;
            this._gapSplit = Math.min(1, Math.max(0, (pos[0] - r.x) / r.w));
            this.setDirtyCanvas(true, false);
        };

        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos, canvas) {
            if (onMouseDown?.apply(this, arguments)) return true;
            const r = this._gapRect();
            if (r && pos[0] >= r.x && pos[0] <= r.x + r.w &&
                pos[1] >= r.y && pos[1] <= r.y + r.h) {
                this._gapDragging = true;
                this.captureInput?.(true);
                this._gapSetSplit(pos);
                return true; // consume so the node is not dragged
            }
            return false;
        };

        const onMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, pos, canvas) {
            onMouseMove?.apply(this, arguments);
            if (this._gapDragging) this._gapSetSplit(pos);
        };

        const onMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (e, pos, canvas) {
            onMouseUp?.apply(this, arguments);
            if (this._gapDragging) {
                this._gapDragging = false;
                this.captureInput?.(false);
            }
        };

        // sensible default size
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.size = [520, 460];
        };
    },
});
