import os
from pdf_markdown_converter import PdfToMarkdownConverter

def main():

    print(os.getcwd())

    converter = PdfToMarkdownConverter(
        input_dir="data/PDF",
        output_dir="data/markdown",
    )

    summary = converter.convert_directory()

    print(summary.total_files)
    print(summary.converted_files)
    print(summary.failed_files)

if __name__ == "__main__":
    
    main()