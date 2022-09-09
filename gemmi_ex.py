import sys
from gemmi import cif
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sympy import sympify

#visualize coord points using matplotlib
#label Ba, Ti, S points as diff colors
#
#verify w/ boyang that it is the unit cell he wants
#replicate unit cell 

def main():
    try:
        doc = cif.read_file('/Users/clairewu/Downloads/gemmi/BTS_Plate_300K_P63cm.cif')  # copy all the data from mmCIF file
        block = doc.sole_block()  # CIF has exactly one block
        coords_by_element = np.array([
            get_plane('_atom_site_fract_x', '_cell_length_a', block),
            get_plane('_atom_site_fract_y', '_cell_length_b', block),
            get_plane('_atom_site_fract_z', '_cell_length_c', block)
        ]).transpose()
        # [Ba_1 Ti_1 Ti_2 S_1 S_2]
        #print(point_coords.transpose()) 
        translation_mat = get_translation_mat('_space_group_symop_operation_xyz', block)
        unit_cell = get_unit_cell_coord(translation_mat, coords_by_element)
        print(coords_by_element[-1])
        print(translation_mat)
        print()
        print(unit_cell[-1])

    except Exception as e:
        print("Oops.", e)
        sys.exit(1)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    #ax.plot3D(xline, yline, zline, 'gray')

    # Data for three-dimensional scattered points
    zdata = coords_by_element[2, :]
    xdata = coords_by_element[0, :]
    ydata = coords_by_element[1, :]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis')
    #plt.show()

#separate by element coord when calling fn
def get_unit_cell_coord(translation_mat, xyz_mat):
    unit_cell = []
    for element in xyz_mat:
        unit_cell.append(apply_translation(translation_mat, element))
    return np.array(unit_cell)

def apply_translation(translation_mat, xyz):
    def map_xyz(x,y,z):
        return {'x': x, 'y': y, 'z': z}
    def compute_translation(coord, t):
        expr = sympify(t)
        return expr.evalf(subs=(map_xyz(*coord)))

    return [
        [compute_translation(xyz, t) for t in translation]
        for translation in translation_mat
    ]
        
    

def get_translation_mat(symm_operations, block):
    translation_expressions = [cif.as_string(i) for i in (block.find_loop(symm_operations))]
    translation_expressions_per_axis = [ 
        [coord_expr.strip() for coord_expr in expr.split(',')] 
        for expr in translation_expressions
    ]
    return np.array(translation_expressions_per_axis)

def get_plane(site_fract_name, cell_length_name, block):
    fract_vals = [as_number(val) for val in block.find_loop(site_fract_name)]
    cell_length = as_number(block.find_value(cell_length_name))
    return [(i * cell_length) for i in fract_vals]

def as_number(num_str):
    return cif.as_number(remove_stddev(num_str))

def remove_stddev(num_str):
    index_of_stddev = num_str.find('(')
    if index_of_stddev >= 0:
        return num_str[:index_of_stddev]
    return num_str #string

if __name__ == '__main__':
    main()