from PIL import Image as PI
import io
import pdf2image
import pyocr
import pyocr.builders
import re

class PDFParser():
    @staticmethod
    def pdf_from_path(
        pdf_path: str
    ) -> str:
        images = pdf2image.convert_from_path(pdf_path)
        img_secondpage = images[1]

        width, height = img_secondpage.size
        left, top, right, bottom = 0, height/5*3, width, height/10*9

        tool = pyocr.get_available_tools()[0]
        lang = 'eng'

        cropped_image = img_secondpage.crop((left, top, right, bottom))
        pdf_text = tool.image_to_string(
            cropped_image,
            lang=lang,
            builder=pyocr.builders.TextBuilder()
        )
        output_text = PDFParser.clean_output(pdf_text)
        return output_text

    @staticmethod
    def clean_output(
        text:str, 
        keep_newlines=False
    ) -> str:
        text = text[text.find('Conventional Mode') + len('Conventional Mode'):]
        if keep_newlines:
            text = re.sub('\n', '', text)
        return text.strip()