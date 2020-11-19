def get_tiff_data(path):
    # import os.path as path
    from PIL import Image
    import pytesseract
    import numpy as np
    import pandas as pd
    from pathlib import Path
    results = []
    # ------------------------------------------------specify Pixel Size = None------------------------------------
    # ------------------------------------------------install software name Tesseract-OCR--------------------------
    Image.MAX_IMAGE_PIXELS = None
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # -------------------------------------------------------------------------------------------------------------
    for metadata_storage_path in path:
        try:

                test = Image.open(metadata_storage_path)
                testarray = np.array(test)
                content = pytesseract.image_to_string(Image.fromarray(testarray))
                r = (metadata_storage_path, content)
                results.append(r)
                print(results)
        except Exception as e:
            message = "MAJOR ERROR DETECTED ON PDF Data FRAME"

    ########################################PDF file DataFrame df_pdf Creation########################################
    df_tiff = pd.DataFrame(results, columns=['metadata_storage_path', 'content'])
    df_tiff['metadata_storage_name'] = df_tiff['metadata_storage_path'].apply(lambda path: Path(path).name)
    df_tiff['metadata_storage_path'] = df_tiff['metadata_storage_path'].apply(
        lambda path: "\\".join(path.split("\\")[0:-1]))
    # print(df_tiff['metadata_storage_path'])
    return df_tiff
file = 'C:\\a_work\\a_iot\\CV_selection_NLP\\appication_analyst_SP_Rec\\Harrigan-Daniel.pdf'
text = get_tiff_data(file)
print(text)






# import tika
#
# file = 'path/to/file'
# # Parse data from file
# file_data = parser.from_file(file)
# # Get files text content
# text = file_data['content']
# print(text)
