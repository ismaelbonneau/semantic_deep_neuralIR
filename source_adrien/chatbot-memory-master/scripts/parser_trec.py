import re


class stringProcessing:
    def __init__(self, text):
        self.text = text

    def removeHTMLTags(self):
        p = re.compile(r'<.*?>')
        return p.sub(' ', self.text)

def clean(to_clean):
    text_cleaned = re.sub("<!-- (.+?) -->", "", ''.join(to_clean))
    text_cleaned = re.sub("`", "", text_cleaned)
    text_cleaned = re.sub("'", "", text_cleaned)
    text_cleaned = re.sub("\"", "", text_cleaned)
    text_cleaned = re.sub("'", "", text_cleaned)
    text_cleaned = re.sub("  ", "", text_cleaned)

    #pas sure que tout ça soit nécessaire
    text_cleaned = re.sub("\\udcc6","",text_cleaned)
    text_cleaned = re.sub("\\udce6", "", text_cleaned)
    text_cleaned = re.sub("\\udcc5", "", text_cleaned)
    text_cleaned = re.sub("\\udceb", "", text_cleaned)
    text_cleaned = re.sub("\\udce3", "", text_cleaned)
    text_cleaned = re.sub("\\udce3", "", text_cleaned)
    text_cleaned = re.sub("\\udcf8", "", text_cleaned)
    text_cleaned = re.sub("\\udcec", "", text_cleaned)


    #text_cleaned = text_cleaned.split('[Text]', 1)[-1]
    #sP = stringProcessing(text_cleaned)
    #text_cleaned = sP.removeHTMLTags()
    return text_cleaned