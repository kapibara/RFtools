import os


def createIF(inputpath, foutputfile):
    letters = os.listdir(inputpath)

    out = open(foutputfile, 'w')

    for l in letters:

        files = os.listdir(inputpath + '/' + l)

        for f in files:
            filename = inputpath + '/' + l + '/' + f

            out.write(filename + ',' + l + '\n')

    out.close()

