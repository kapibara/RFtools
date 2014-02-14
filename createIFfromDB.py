import os
import glob
import cv2

def createIF(inputpath, foutputfile):
    letters = os.listdir(inputpath)

    out = open(foutputfile, 'w')
    fc = 0

    for l in letters:

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

def createMaskedIF(inputpath, foutputfile):
    letters = os.listdir(inputpath)

    out = open(foutputfile, 'w')
    fc = 0

    for l in letters:

#        if (l=='m') or (l=='n'):

            files = glob.glob(inputpath + '/' + l + '/imDepthOrig*')

            for f in files:
                maskname = f.replace('imDepthOrig','imDepthMask')

                I= cv2.imread(f,-1)
                Im = cv2.imread(maskname,-1)/255

                I = I*Im

                newname = f.replace('imDepthOrig','imDepthOM')

                cv2.imwrite(newname,I)

                newname = newname.replace('\\', '/')

                if fc < 1000000:

                    out.write(newname + ',' + l + '\n')
                else:

                    out.close()

                    print 'files read: {0}'.format(fc)

                    return

                fc = fc + 1

    out.close()

    print 'files read: {0}'.format(fc)
