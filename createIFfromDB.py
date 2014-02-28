import os
import glob
import cv2
import re

def createIF(inputpath, foutputfile, classes = 'a-z'):
    letters = os.listdir(inputpath)

    out = open(foutputfile, 'w')
    fc = 0
    
    pattern = '^[' + classes + ']'
    
    lmatcher = re.compile(pattern)

    pattern = '^[' + classes + ']'

    lmatcher = re.compile(pattern)

    for l in letters:
      
        if lmatcher.match(l):

	  files = glob.glob(inputpath + '/' + l + '/imDepthOrig*')

	  for f in files:

	      f = f.replace('\\', '/')

	      if fc < 1000:

		  out.write(f + ',' + l + '\n')
	      else:

		  out.close()

		  print 'files read: {0}'.format(fc)

		  return

	      fc = fc + 1

    out.close()

    print 'files read: {0}'.format(fc)

def createMaskedIF(inputpath, foutputfile, cl = 'a-z', maximcount = 100000):
    letters = os.listdir(inputpath)
    
    pattern = '^[' + cl + ']'
    
    lmatcher = re.compile(pattern)

    pattern = '^[' + cl + ']'

    lmatcher = re.compile(pattern)

    out = open(foutputfile, 'w')
    fc = 0

    for l in letters:

        if lmatcher.match(l):
            files = glob.glob(inputpath + '/' + l + '/imDepthOrig*')
            
            if len(files) > maximcount:
                files = files[0:maximcount]

            if len(files) > maximcount:
                files = files[0:maximcount]

            for f in files:
                maskname = f.replace('imDepthOrig','imDepthMask')

                I= cv2.imread(f,-1)
                Im = cv2.imread(maskname,-1)/255

                I = I*Im

                newname = f.replace('imDepthOrig','imDepthOM')

                cv2.imwrite(newname,I)

                newname = newname.replace('\\', '/')

                if fc < 10000:

                    out.write(newname + ',' + l + '\n')
                else:

                    out.close()

                    print 'files read: {0}'.format(fc)

                    return

                fc = fc + 1

    out.close()

    print 'files read: {0}'.format(fc)
