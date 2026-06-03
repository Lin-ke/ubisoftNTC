import sys
import os
import pymupdf4llm

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/process.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join("assets", pdf_path)

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    md_text = pymupdf4llm.to_markdown(pdf_path)

    output_path = os.path.join("reference", os.path.splitext(os.path.basename(pdf_path))[0] + ".md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    main()
