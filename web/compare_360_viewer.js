// GAP 360 Compare Viewer - interactive 360° equirectangular before/after split viewer
// by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
//
// Interactive equirectangular panorama viewer drawn directly on the LiteGraph node canvas.
// Supports click-and-drag 360° camera rotation (yaw & pitch), zoom (FOV), and split-slider comparison.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PAD = 10;
const TOP = 54;

function imgUrl(info) {
    return api.apiURL(
        `/view?filename=${encodeURIComponent(info.filename)}` +
        `&type=${info.type}&subfolder=${encodeURIComponent(info.subfolder || "")}` +
        `&rand=${Math.random()}`);
}

app.registerExtension({
    name: "geekatplay.genupscale.compare_360_viewer",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GAP360CompareViewer") return;

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
            if (this._gapYaw === undefined) this._gapYaw = 0;
            if (this._gapPitch === undefined) this._gapPitch = 0;
            if (this._gapFov === undefined) this._gapFov = 75;
            if (this._gapSplit === undefined) this._gapSplit = 0.5;
            this.setDirtyCanvas(true, false);
        };

        nodeType.prototype._gapRect = function () {
            const w = this.size[0] - PAD * 2;
            const h = this.size[1] - TOP - PAD;
            if (w < 20 || h < 20) return null;
            return { x: PAD, y: TOP, w, h };
        };

        // Render 360 equirectangular perspective view onto canvas context
        nodeType.prototype._draw360Panorama = function (ctx, r, img, clipLeft, clipRight) {
            if (!img || !img.naturalWidth || !img.naturalHeight) return;

            const yaw = (this._gapYaw ?? 0) % 360;
            const pitch = Math.min(80, Math.max(-80, this._gapPitch ?? 0));
            const fov = this._gapFov ?? 75;

            ctx.save();
            ctx.beginPath();
            ctx.rect(clipLeft, r.y, clipRight - clipLeft, r.h);
            ctx.clip();

            // Equirectangular viewport mapping
            const iw = img.naturalWidth;
            const ih = img.naturalHeight;

            // Crop width corresponding to current FOV in 360 degrees
            const srcW = (fov / 360.0) * iw;
            const srcH = srcW * (r.h / r.w);

            // Center longitude based on yaw (0..360)
            let srcX = ((yaw / 360.0) * iw) - (srcW / 2);
            // Center latitude based on pitch (-90..90)
            let srcY = ((0.5 - (pitch / 180.0)) * ih) - (srcH / 2);
            srcY = Math.min(ih - srcH, Math.max(0, srcY));

            // Wrap horizontal panning smoothly across 360 boundary
            srcX = (srcX % iw + iw) % iw;

            if (srcX + srcW <= iw) {
                ctx.drawImage(img, srcX, srcY, srcW, srcH, r.x, r.y, r.w, r.h);
            } else {
                // Draw wrapped left and right segments
                const w1 = iw - srcX;
                const dw1 = (w1 / srcW) * r.w;
                ctx.drawImage(img, srcX, srcY, w1, srcH, r.x, r.y, dw1, r.h);
                ctx.drawImage(img, 0, srcY, srcW - w1, srcH, r.x + dw1, r.y, r.w - dw1, r.h);
            }

            ctx.restore();
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags.collapsed) return;
            const r = this._gapRect();
            if (!r) return;

            if (!this._gapA && !this._gapB) {
                ctx.fillStyle = "#888";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Run workflow to load interactive 360° comparison view",
                    this.size[0] / 2, TOP + 40);
                return;
            }

            const split = this._gapSplit ?? 0.5;
            const sx = r.x + r.w * split;

            // Draw AFTER (b) on right side of split, BEFORE (a) on left side
            this._draw360Panorama(ctx, r, this._gapB, sx, r.x + r.w);
            this._draw360Panorama(ctx, r, this._gapA, r.x, sx);

            // Draw 360 Split Divider line
            ctx.strokeStyle = "#00e5ff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(sx, r.y);
            ctx.lineTo(sx, r.y + r.h);
            ctx.stroke();

            // Split grab handle
            const cy = r.y + r.h / 2;
            ctx.fillStyle = "rgba(0, 229, 255, 0.95)";
            ctx.beginPath();
            ctx.arc(sx, cy, 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = "#111";
            ctx.beginPath();
            ctx.moveTo(sx - 7, cy); ctx.lineTo(sx - 2, cy - 4); ctx.lineTo(sx - 2, cy + 4);
            ctx.moveTo(sx + 7, cy); ctx.lineTo(sx + 2, cy - 4); ctx.lineTo(sx + 2, cy + 4);
            ctx.fill();

            // Overlay Badges & Controls Info
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "left";
            ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
            ctx.fillRect(r.x + 6, r.y + 6, 82, 18);
            ctx.fillStyle = "#00e5ff";
            ctx.fillText("360° BEFORE", r.x + 10, r.y + 19);

            ctx.textAlign = "right";
            ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
            ctx.fillRect(r.x + r.w - 78, r.y + 6, 72, 18);
            ctx.fillStyle = "#00ffff";
            ctx.fillText("360° AFTER", r.x + r.w - 10, r.y + 19);

            // Navigation hints at bottom
            ctx.font = "10px Arial";
            ctx.textAlign = "center";
            ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
            ctx.fillRect(r.x + (r.w / 2) - 100, r.y + r.h - 20, 200, 16);
            ctx.fillStyle = "#fff";
            ctx.fillText(`Drag to rotate 360° | Yaw: ${Math.round(this._gapYaw ?? 0)}° Pitch: ${Math.round(this._gapPitch ?? 0)}°`,
                r.x + r.w / 2, r.y + r.h - 8);
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
                const sx = r.x + r.w * (this._gapSplit ?? 0.5);
                if (Math.abs(pos[0] - sx) < 15) {
                    this._gapDraggingSplit = true;
                    this._gapSetSplit(pos);
                } else {
                    this._gapDragging360 = true;
                    this._gapLastMouse = [pos[0], pos[1]];
                }
                this.captureInput?.(true);
                return true;
            }
            return false;
        };

        const onMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, pos, canvas) {
            onMouseMove?.apply(this, arguments);
            if (this._gapDraggingSplit) {
                this._gapSetSplit(pos);
            } else if (this._gapDragging360 && this._gapLastMouse) {
                const dx = pos[0] - this._gapLastMouse[0];
                const dy = pos[1] - this._gapLastMouse[1];
                this._gapLastMouse = [pos[0], pos[1]];

                const sens = 0.4;
                this._gapYaw = ((this._gapYaw ?? 0) - dx * sens) % 360;
                this._gapPitch = Math.min(80, Math.max(-80, (this._gapPitch ?? 0) + dy * sens));
                this.setDirtyCanvas(true, false);
            }
        };

        const onMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (e, pos, canvas) {
            onMouseUp?.apply(this, arguments);
            if (this._gapDraggingSplit || this._gapDragging360) {
                this._gapDraggingSplit = false;
                this._gapDragging360 = false;
                this.captureInput?.(false);
            }
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.size = [560, 480];
        };
    },
});
