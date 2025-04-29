from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np

def text_link(pdf, text, links, y, left_margin, right_margin, fontsize):
    current_x = left_margin
    current_y = y
    usable_width = pdf._pagesize[0] - left_margin - right_margin
    line_height = fontsize + 2

    def draw_word(word, is_link=False, url=None):
        nonlocal current_x, current_y
        word_width = pdf.stringWidth(word)

        if current_x + word_width > left_margin + usable_width:
            current_x = left_margin
            current_y -= line_height

        if is_link:
            pdf.setFillColorRGB(0, 0, 1)
            pdf.drawString(current_x, current_y, word)
            pdf.setStrokeColorRGB(0, 0, 1)
            pdf.line(current_x, current_y - 1, current_x + word_width, current_y - 1)
            if url:
                pdf.linkURL(url, (current_x, current_y - 2, current_x + word_width, current_y + 12))
        else:
            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawString(current_x, current_y, word)

        current_x += word_width + pdf.stringWidth(" ")

    link_dict = {phrase: url for phrase, url in links}
    link_phrases = sorted(link_dict.keys(), key=lambda x: -len(x.split()))

    words = text.split()
    i = 0

    while i < len(words):
        matched = False
        for phrase in link_phrases:
            phrase_words = phrase.split()
            if words[i:i + len(phrase_words)] == phrase_words:
                draw_word(' '.join(phrase_words), is_link=True, url=link_dict[phrase])
                i += len(phrase_words)
                matched = True
                break
        if not matched:
            draw_word(words[i])
            i += 1

    pdf.setFillColorRGB(0, 0, 0)
    return pdf

def generate_pdf_summary(data=None, affine=None, raw=None, json_data=None, output_path="README.pdf"):
    page_width, page_height = letter
    left_margin = 50
    right_margin = 50
    bottom_margin = 50
    top_margin = 50
    fontsize = 12
    title_fontsize = 20
    usable_width = page_width - left_margin - right_margin

    pdf = canvas.Canvas(output_path, pagesize=letter)
    y = page_height - top_margin

    if data is not None:
        extension = ".nii.gz"
    elif raw is not None:
        extension = ".edf"
    else:
        extension = "unknown"

    supported_exts = (".nii.gz", ".json", ".tsv", ".edf")

    if extension not in supported_exts:
        pdf.setFont("Helvetica-Bold", 30)
        pdf.drawString(left_margin, y-10, "WARNING")
        y -= 50

        pdf.setFont("Helvetica", fontsize)
        text_1 = "Unrecognized extension. The input file must be .nii.gz, .json, .tsv, or .edf."
        pdf = text_link(pdf, text_1, [], y, left_margin, right_margin, fontsize)
    else:
        if extension == ".nii.gz":
            modality = 'fMRI'
            library = 'nibabel'
            library_link = "https://nipy.org/nibabel/"
            packages_links = [("nilearn", "https://pypi.org/project/nilearn/"),
                              ("fMRIPrep", "https://pypi.org/project/fmriprep/")]
        else:
            modality = 'EEG'
            library = 'MNE'
            library_link = "https://mne.tools/stable/install/index.html"
            packages_links = [("PyEEG", "https://pypi.org/project/pyeeg/")]

        pdf.setFont("Helvetica-Bold", 30)
        pdf.drawString(left_margin, y-10, modality)
        y -= 20

        pdf.line(left_margin, y, page_width - right_margin, y)
        y -= 50

        pdf.setFont("Helvetica", fontsize)
        text_1 = f"This is a {modality} image. To preprocess these images we recommend installing {library} library."
        links = [(library, library_link)]
        pdf = text_link(pdf, text_1, links, y, left_margin, right_margin, fontsize)
        y -= 50

        pdf.setFont("Helvetica-Bold", title_fontsize)
        pdf.drawString(left_margin, y, "Quality Control")
        y -= 20

        pdf.setFont("Helvetica", fontsize)
        if extension == ".nii.gz":
            text_2 = "Additionally, to automate quality control (QC), you may consider nilearn and fMRIPrep packages."
        else:
            text_2 = "Additionally, to automate quality control (QC), you may consider PyEEG package."

        pdf = text_link(pdf, text_2, packages_links, y, left_margin, right_margin, fontsize)
        y -= 50

        pdf.setFont("Helvetica-Bold", title_fontsize)
        pdf.drawString(left_margin, y, "Warnings/Recommendations")
        y -= 20

        pdf.setFont("Helvetica", fontsize)
        snr = 60
        if snr > 50:
            text_3 = f"The signal-to-noise ratio (SNR) is {snr}. Images with this SNR are sharp and well-contrasted."
        elif snr > 10:
            text_3 = f"The signal-to-noise ratio (SNR) is {snr}. Acceptable but more susceptible to noise."
        else:
            text_3 = f"The signal-to-noise ratio (SNR) is {snr}. This SNR is generally considered unacceptable."

        pdf = text_link(pdf, text_3, [], y, left_margin, right_margin, fontsize)
        y -= 50

        pdf.setFont("Helvetica-Bold", title_fontsize)
        pdf.drawString(left_margin, y, "Visualization")
        y -= 20

        pdf.setFont("Helvetica", fontsize)
        text_4 = "This plot represents ... The y-axis represents ..., the x-axis represents ..."
        pdf = text_link(pdf, text_4, [], y, left_margin, right_margin, fontsize)
        y -= 50

    pdf.save()
