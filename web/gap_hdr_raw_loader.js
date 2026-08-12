// GAP Load EXR / DNG / HDR Image - Frontend Extension for RAW & HDR File Browser
// by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
//
// Solves HTTP 413 (Content Too Large) errors on large 360 DNG files by supporting direct local path resolution,
// custom file extension filters (.dng, .exr, .hdr, .tiff, .raw), and direct file selection.

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

                // Handle large files (>25MB) to avoid HTTP 413 Content Too Large server errors
                const isLarge = file.size > 25 * 1024 * 1024;
                const pathWidget = this.widgets?.find(w => w.name === "custom_file_path");

                if (isLarge) {
                    // Try setting direct path if available from webkitRelativePath or path property
                    const directPath = file.path || file.name;
                    if (pathWidget) {
                        pathWidget.value = directPath;
                    }
                    alert(`Large DNG/EXR file selected (${(file.size / (1024 * 1024)).toFixed(1)} MB).\n\n` +
                          `To prevent HTTP 413 (Content Too Large) server errors:\n` +
                          `1. Copy the file into your 'ComfyUI/input/' folder, OR\n` +
                          `2. Paste the full file path into 'custom_file_path'.\n\n` +
                          `Set file path: ${directPath}`);
                    this.setDirtyCanvas(true, false);
                    return;
                }

                // For standard size files, use standard ComfyUI upload API
                try {
                    const formData = new FormData();
                    formData.append("image", file);
                    formData.append("overwrite", "true");

                    const response = await fetch("/upload/image", {
                        method: "POST",
                        body: formData,
                    });

                    if (response.status === 413) {
                        alert("HTTP 413: Content Too Large.\n\n" +
                              "Please paste the full absolute file path into the 'custom_file_path' widget instead of uploading via HTTP.");
                        return;
                    }

                    if (!response.ok) {
                        throw new Error(`Upload failed: ${response.statusText}`);
                    }

                    const data = await response.json();
                    const comboWidget = this.widgets?.find(w => w.name === "image_file");
                    if (comboWidget) {
                        if (!comboWidget.options.values.includes(data.name)) {
                            comboWidget.options.values.push(data.name);
                        }
                        comboWidget.value = data.name;
                    }
                    this.setDirtyCanvas(true, false);
                } catch (err) {
                    console.error("GAP HDR Load Upload Error:", err);
                }
            });

            // Add 'Upload / Browse RAW (DNG/EXR)' widget button
            this.addWidget("button", "📁 Browse DNG / EXR / RAW", null, () => {
                fileInput.click();
            });

            this.size = [420, 240];
        };
    },
});
