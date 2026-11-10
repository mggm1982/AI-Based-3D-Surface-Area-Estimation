"""
Turn the Beamer PDF into a .pptx with one full-slide image per page.

  python slides/make_pptx.py                                  -> slides/Project51_ManasJain.pptx
  python slides/make_pptx.py --notes notes.json --out x.pptx  -> same, with speaker notes (keys s01, s02, ...)

Needs pdftoppm (poppler) and python-pptx.
"""
import argparse, glob, json, os, subprocess, tempfile
from pptx import Presentation
from pptx.util import Emu

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", default=os.path.join("slides", "beamer", "slides.pdf"))
ap.add_argument("--out", default=os.path.join("slides", "Project51_ManasJain.pptx"))
ap.add_argument("--notes", default=None)
args = ap.parse_args()

notes = json.load(open(args.notes)) if args.notes else {}
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["pdftoppm", "-png", "-scale-to-x", "2560", "-scale-to-y", "1440", args.pdf,
                    os.path.join(tmp, "p")], check=True)
    pages = sorted(glob.glob(os.path.join(tmp, "p-*.png")))
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 16:9
    blank = prs.slide_layouts[6]
    for i, png in enumerate(pages, 1):
        s = prs.slides.add_slide(blank)
        s.shapes.add_picture(png, 0, 0, width=prs.slide_width, height=prs.slide_height)
        key = f"s{i:02d}"
        if key in notes:
            s.notes_slide.notes_text_frame.text = notes[key]
    prs.save(args.out)
print("wrote", args.out, f"({len(pages)} slides)")
