import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import _pickle as pk


def plot_mfcc_overlapped(ax, mfcc, orc_bnds, uns_bnds):
    cax = ax.imshow(mfcc.T, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    for bnd in orc_bnds:
        cax = ax.axvline(x=bnd, color='red', linestyle=':')
    for bnd in uns_bnds:
        if bnd in orc_bnds:
            if bnd - 1 >= 0:
                bnd = bnd - 1
            else:
                bnd = bnd + 1
        cax = ax.axvline(x=bnd)
    ax.set_aspect(aspect=0.4)

def plot_mfcc(ax, mfcc, bnds, **kwargs):
    cax = ax.imshow(mfcc.T, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    for bnd in bnds:
        cax = ax.axvline(x=bnd, **kwargs)

def read_pickle(path):
    return pk.load(open(path,'rb'))
def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o' , '--orc_boundary_path' , type=str, default='', help='')
    parser.add_argument('-u1', '--uns1_boundary_path', type=str, default='', help='')
    parser.add_argument('-u2', '--uns2_boundary_path', type=str, default='', help='')
    parser.add_argument('-u3', '--uns3_boundary_path', type=str, default='', help='')
    parser.add_argument('-m', '--mfcc_path', type=str, default='', help='')
    parser.add_argument('-i', '--index', type=int, help='')
    parser.add_argument('-s', '--save_path', type=str, default='./bnd_visualization.png', help='')

    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    orc = read_pickle(args.orc_boundary_path)[args.index][1:-1]
    uns1 = read_pickle(args.uns1_boundary_path)[args.index][1:-1]
    uns2 = read_pickle(args.uns2_boundary_path)[args.index][1:-1]
    uns3 = read_pickle(args.uns3_boundary_path)[args.index][1:-1]
    mfcc = read_pickle(args.mfcc_path)[args.index]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, tight_layout=True)
    plot_mfcc_overlapped(ax1, mfcc, orc, uns1)
    plot_mfcc_overlapped(ax2, mfcc, orc, uns2)
    plot_mfcc_overlapped(ax3, mfcc, orc, uns3)
    

    '''
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
    plot_mfcc(ax0, mfcc, orc, linestyle='-')
    plot_mfcc(ax1, mfcc, uns1)
    plot_mfcc(ax2, mfcc, uns2)
    plot_mfcc(ax3, mfcc, uns3)
    '''
    plt.tight_layout(pad=-1)
    plt.savefig(args.save_path)
    #plt.show()




