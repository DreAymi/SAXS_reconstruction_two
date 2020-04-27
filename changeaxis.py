import os
import numpy as np
import multiprocessing


def read_pdb_xyz(pdb_name):
    xyz = []
    with open(pdb_name, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # extract x, y, z coordinates for carbon alpha atoms
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                if "CA" in line[12:16].strip() or "C3" in line[12:16].strip() or "C5" in line[12:16].strip() or "P"==line[12:16].strip():
                    xyz.append([x, y, z])
    return xyz


def getprincipalaxis(pdb_name):
    xyz=read_pdb_xyz(pdb_name)
    coord = np.array(xyz, float)
    center = np.mean(coord, 0)
    coord = coord - center
    # compute principal axis matrix
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    order = np.argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()
    return axis1,axis2,axis3,center


def change_to_new_axis(pdb_name):

    pdb_name=pdb_name.strip()
    try:
        filea=open('rotate_pisa_pdb/%s'%pdb_name,'a')
        lines=[]
        wholename='pisa_pdb/%s'%pdb_name
        
        axis1,axis2,axis3,center=getprincipalaxis(wholename)
        a=np.array([[axis1[0],axis2[0],axis3[0]],[axis1[1],axis2[1],axis3[1]],[axis1[2],axis2[2],axis3[2]]])
        with open(wholename, 'r') as pdb_file:
            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # extract x, y, z coordinates for carbon alpha atoms
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    b = np.array([[x-center[0]],[y-center[0]],[z-center[0]]])
                    newpoint=np.linalg.solve(a,b).reshape(-1)
                    strx='{:>8.3f}'.format(newpoint[0])
                    stry='{:>8.3f}'.format(newpoint[1])
                    strz='{:>8.3f}'.format(newpoint[2])
                    lines.append(line[:30]+strx+stry+strz+line[54:])
                else:
                    lines.append(line)
            filea.writelines(lines)
        filea.close()
    except:
        print "%s error!!!!!!!!!!!!!!!!!!"%pdb_name

def multi_run():
    f=open('pdbname.txt')
    pdbnames=f.readlines()
    f.close()
    total=len(pdbnames)
    pdbnames=np.array(pdbnames)
    pdbnames=np.sort(pdbnames)
    
    pool=multiprocessing.Pool(processes=5)
    pool.map(change_to_new_axis,pdbnames)
    pool.close()
    pool.join()
    


if __name__=='__main__':
    multi_run()
