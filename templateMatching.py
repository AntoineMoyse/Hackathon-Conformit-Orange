import cv2 as cv
import numpy as np
import sys
import copy
import pytesseract
from os import listdir
from pdf2image import convert_from_path


def main(argv, arc):
    if (arc<2):return
    
    #Convert to PDF
    print("Converting to PDF", end = ' ')
    img_base = getImageFromPDF(argv[1])
    img_output = copy.deepcopy(img_base)
    print(": Done\n")
    
    
    #recover templates from folder
    print("Recovering Templates")
    templates = loadTemplates('templates/')
    print("Done\n")
    
    
    #extract templates from image
    print("Extracting Symboles")
    rectangles = [];i=0
    for temp in templates:
        rectangle = figureDetect(img_base, temp)
        rectangles += rectangle
        i+=1
        print('{0}/{1}\r'.format(i, len(templates)), end='')
    print("\nDone\n")
    
    
    #save only figs representation
    img_figs = copy.deepcopy(img_base)
    for pt in rectangles:
        cv.rectangle(img_figs, pt[0], (pt[1] , pt[2]), (0,0,255), 2)
    cv.imwrite('out/2_figs_only.png',img_figs)
    
    
    #remove figs from original file
    img_nofigs = copy.deepcopy(img_base)
    for pt in rectangles:
        cv.rectangle(img_nofigs, pt[0], (pt[1] , pt[2]), (255,255,255), -1)
    cv.imwrite('out/3_nofigs.png',img_nofigs)
    
    
    #extract text
    print("Extracting Texts")
    img_nofigs_notext = copy.deepcopy(img_nofigs)
    data = pytesseract.image_to_data(img_nofigs, output_type='dict')
    boxes = len(data['level'])
    size = len(range(boxes))
    for i in range(boxes):
        (x,y,w,h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        if(data['text'][i] == '' or data['text'][i] == '-' or data['text'][i] == '|' or data['text'][i] == 'l' or data['text'][i] == 'T' or data['text'][i] == '<'
                or data['text'][i] == '>' or data['text'][i] == '><' or data['text'][i] == ' ' or data['text'][i] == '  ' or data['text'][i] == '   '
                or data['text'][i] == '    ' or data['text'][i] == '     '):
            i=i+1
            print('{0}/{1}\r'.format(i, size), end='')
        else:
                cv.rectangle(img_nofigs_notext, (x, y), (x + w, y + h), (255,255,255), -1)
                cv.rectangle(img_output, (x, y), (x + w, y + h), (255,0,0), 3)

    #save text datas
    cv.imwrite('out/4_text_only.png',img_output)
    cv.imwrite('out/5_nofigsnotext.png',img_nofigs_notext)
    #Scan l'image entière pour du texte et écrit son contenu dans un txt
    touttexte=pytesseract.image_to_string(img_nofigs)
    with open('out/texts.txt', mode = 'w') as f:
        f.write(touttexte)
    print("\nDone")
    print()
    
    
    #extract lines from image
    print("Extracting lignes", end=' ')
    #img_nofigs_notext = cv.imread(argv[2])
    lines = lineDetect(img_nofigs_notext)
    print(': Done')
    print()
    
    
    #draw lines
    img_lines = copy.deepcopy(img_base)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img_lines,(x1,y1),(x2,y2),(0,255,0),2)
    cv.imwrite('out/6_lines_only.png',img_lines)
    
    
    #draw final image
    print("Drawing Final", end=' ')
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img_output,(x1,y1),(x2,y2),(0,255,0),2)
    for pt in rectangles:
        cv.rectangle(img_output, pt[0], (pt[1] , pt[2]), (0,0,255), 2)
    cv.imwrite('out/7_output.png',img_output)
    print(': Done')
    print()

    
def getImageFromPDF(file):
    images = convert_from_path(file)
    images[0].save('out/1_png_basefile.png', 'PNG')
    return cv.imread('out/1_png_basefile.png')
    
def loadTemplates(folder):
    ret = []
    for file in listdir(folder):
        print('  • '+file)
        template = cv.imread(folder + file,0)
        
        North = cv.imread(folder + file,0)
        South = cv.rotate(North, cv.ROTATE_90_CLOCKWISE)
        East = cv.rotate(South, cv.ROTATE_90_CLOCKWISE)
        West = cv.rotate(East, cv.ROTATE_90_CLOCKWISE)
        
        NorthFlipped = cv.flip(North, 0)
        SouthFlipped = cv.flip(South, 0)
        EastFlipped = cv.flip(East, 0)
        WestFlipped = cv.flip(West, 0)
        
        ret.extend([North, South, East, West, NorthFlipped, SouthFlipped, EastFlipped, WestFlipped])
    return ret

def figureDetect(img, template):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    
    ret = []
    for pt in zip(*loc[::-1]):
        p1 = pt
        p2 = pt[0] + w
        p3 = pt[1] + h
        pa = [p1, p2, p3]
        ret.append(pa)
        
    return ret

def lineDetect(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    
    lines = cv.HoughLinesP(edges, rho=0.5, theta=0.1 * np.pi / 180, threshold=25, minLineLength=5, maxLineGap=15)
    
    return lines

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
