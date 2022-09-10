import sys
from gemmi import cif
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sympy import sympify

#visualize coord points using matplotlib
#label Ba, Ti, S points as diff colors
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

        translation_mat = get_translation_mat('_space_group_symop_operation_xyz', block)
        unit_cell = get_unit_cell_coord(translation_mat, coords_by_element)
        print("xyz=", coords_by_element)
        #print(translation_mat)
        print()
        print(unit_cell[0])

    except Exception as e:
        print("Oops.", e)
        sys.exit(1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    Ba_data = unit_cell[0, :, :]
    Ti_1_data = unit_cell[1, :, :]
    Ti_2_data = unit_cell[2, :, :]
    S_1_data = unit_cell[3, :, :]
    S_2_data = unit_cell[4, :, :]
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    labels = block.find_loop('_atom_site_label')
    print(labels)
    for element, label in zip(unit_cell, labels):
        ax.scatter(element[:, 0], element[:, 1], element[:, 2], label=label)

    # ax.scatter3D(Ba_data, Ti_1_data, Ti_2_data, S_1_data, S_2_data, c=c, cmap='viridis')
    ax.legend()
    plt.show()

def hex_to_cart():
    return
    
#builds matrices for each element's symm operation
def get_unit_cell_coord(translation_mat, coords_by_element):
    unit_cell = []
    for element in coords_by_element:
        symmetry_operation_result = apply_translations(translation_mat, element)
        unit_cell.append(symmetry_operation_result)
    return np.array(unit_cell)

#uses sympy package to assign x,y,z symm op from CIF file to numerical values
def apply_translations(translation_mat, xyz):
    def map_xyz(x,y,z):
        return {'x': x, 'y': y, 'z': z}
    def compute_translation(xyz, t):
        expr = sympify(t)
        return expr.evalf(subs=(map_xyz(*xyz)))

        """
        map_xyz(*xyz) is the same as

        map_xyz(xyz[0], xyz[1], xyz[2])
        """

    return [
        [compute_translation(xyz, t) for t in translation] 
        for translation in translation_mat # t is x, y, z point in row. translation is row (12).
    ]
"""
Example output for S_2
xyz= [9.71108897 3.87698949 1.7026527 ]
[['x' 'y' 'z']
 ['-y' 'x-y' 'z']
 ['-x+y' '-x' 'z']
 ['-x' '-y' 'z+1/2']
 ['y' '-x+y' 'z+1/2']
 ['x-y' 'x' 'z+1/2']
 ['-y' '-x' 'z+1/2']
 ['-x+y' 'y' 'z+1/2']
 ['x' 'x-y' 'z+1/2']
 ['y' 'x' 'z']
 ['x-y' '-y' 'z']
 ['-x' '-x+y' 'z']]

[[9.71108897000000 3.87698949000000 1.70265270000000]
 [-3.87698949000000 5.83409948000000 1.70265270000000]
 [-5.83409948000000 -9.71108897000000 1.70265270000000]
 [-9.71108897000000 -3.87698949000000 2.20265270000000]
 [3.87698949000000 -5.83409948000000 2.20265270000000]
 [5.83409948000000 9.71108897000000 2.20265270000000]
 [-3.87698949000000 -9.71108897000000 2.20265270000000]
 [-5.83409948000000 3.87698949000000 2.20265270000000]
 [9.71108897000000 5.83409948000000 2.20265270000000]
 [3.87698949000000 9.71108897000000 1.70265270000000]
 [5.83409948000000 -3.87698949000000 1.70265270000000]
 [-9.71108897000000 -5.83409948000000 1.70265270000000]]
"""        
    
#extracts symm op from CIF file into a matrix of strings
def get_translation_mat(symm_operations, block):
    translation_expressions = [cif.as_string(i) for i in (block.find_loop(symm_operations))]
    translation_expressions_per_axis = [ 
        [coord_expr.strip() for coord_expr in expr.split(',')] 
        for expr in translation_expressions
    ]
    return np.array(translation_expressions_per_axis)

#extracts fract site values and cell length and multiple for xyz coord
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