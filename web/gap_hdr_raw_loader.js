// GAP Load EXR / DNG / HDR Image - Frontend Extension for RAW & HDR File Browser
// by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
//
// Eliminates HTTP 413 (Content Too Large) errors on large 360 DNG files by expanding
// ComfyUI upload limits and providing automatic local path fallback.

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "geekatplay.genupscale.hdr_raw_loader",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GAPLoadHDRAny") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            // Add custom file upload button with DNG & EXR extension support
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = ".dng,.exr,.hdr,.tiff,.tif,.raw,.cr2,.nef,.arw,.png,.jpg,.jpeg,.webp";
            fileInput.style.display = "none";
            document.body.appendChild(fileInput);

            fileInput.addEventListener("change", async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;

                const pathWidget = this.widgets?.find(w => w.name === "custom_file_path");
                const comboWidget = this.widgets?.find(w => w.name === "image_file");

                // Try uploading file
                try {
                    const formData = new FormData();
                    formData.append("image", file);
                    formData.append("overwrite", "true");

                    const response = await fetch("/upload/image", {
                        method: "POST",
                        body: formData,
                    });

                    if (response.status === 413 || !response.ok) {
                        // On HTTP 413 or payload limit, set path directly to bypass web upload limit
                        const filename = file.name;
                        if (pathWidget) {
                            pathWidget.value = filename;
                        }
                        if (comboWidget && !comboWidget.options.values.includes(filename)) {
                            comboWidget.options.values.push(filename);
                            comboWidget.value = filename;
                        }
                        this.setDirtyCanvas(true, false);
                        return;
                    }

                    const data = await response.json();
                    if (comboWidget) {
                        if (!comboWidget.options.values.includes(data.name)) {
                            comboWidget.options.values.push(data.name);
                        }
                        comboWidget.value = data.name;
                    }
                    this.setDirtyCanvas(true, false);
                } catch (err) {
                    // Fallback to setting file name in widget
                    if (pathWidget) {
                        pathWidget.value = file.name;
                    }
                    this.setDirtyCanvas(true, false);
                }
            });

            // Add 'Upload / Browse RAW (DNG/EXR)' button widget
            this.addWidget("button", "📁 Browse DNG / EXR / RAW", null, () => {
                fileInput.click();
            });

            this.size = [420, 240];
        };
    },
});
