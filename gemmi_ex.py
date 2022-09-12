import sys
from gemmi import cif
import numpy as np
import math
from matplotlib import pyplot as plt
from sympy import Matrix, Symbol

#visualize coord points using matplotlib
#label Ba, Ti, S points as diff colors
#verify w/ boyang that it is the unit cell he wants
#replicate unit cell 

def main():
    doc = cif.read_file('/Users/clairewu/Downloads/gemmi/BTS_Plate_300K_P63cm.cif')  # copy all the data from mmCIF file
    block = doc.sole_block()  # CIF has exactly one block
    cart_coords_by_element = get_cart_coords_by_element(block)
    cart_translation_mat = get_cart_translation_matrix(block)

    print(cart_translation_mat)
    
    unit_cell = get_unit_cell_coord(cart_translation_mat, cart_coords_by_element)
    print("xyz=", cart_coords_by_element)
    print()
    print(unit_cell)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    labels = block.find_loop('_atom_site_label')
    
    for element, label in zip(unit_cell, labels):
        ax.scatter(element[:, 0], element[:, 1], element[:, 2], label=label)
        #plt.show()

    ax.legend()
    plt.show()

#extract atom site fract & cell length from CIF file
def get_cart_coords_by_element(block):
    hex_coords_by_element = np.array([
        get_plane('_atom_site_fract_x', '_cell_length_a', block),
        get_plane('_atom_site_fract_y', '_cell_length_b', block),
        get_plane('_atom_site_fract_z', '_cell_length_c', block)
    ]).transpose()
    # [Ba_1 Ti_1 Ti_2 S_1 S_2]
    return hex_to_cart_np(hex_coords_by_element)

#extract symmetry operations from CIF file & convert to cartesian coordinates
def get_cart_translation_matrix(block):
    hex_translation_mat = get_translation_mat('_space_group_symop_operation_xyz', block)
    cart_translation_mat = hex_to_cart_sympy(hex_translation_mat)
    return cart_translation_mat

# converts hex to cart coordinates np.array
def hex_to_cart_np(hexagonal_coords):
    hex_to_cart_mat = np.array([[1, -.5, 0], [math.sqrt(3)/2, 0, 0], [0, 0, 1]])
    cartesian_coord = np.dot(hex_to_cart_mat, hexagonal_coords.T)
    return cartesian_coord.T

# converts hex to cart coordinates sympy array
def hex_to_cart_sympy(hexagonal_coords):
    hex_to_cart_mat = Matrix([[1, -.5, 0], [math.sqrt(3)/2, 0, 0], [0, 0, 1]])
    cartesian_coords = hex_to_cart_mat * hexagonal_coords.T
    return cartesian_coords.T

#builds matrices for each element's symm operation
def get_unit_cell_coord(translation_mat, coords_by_element):
    unit_cell = []
    for element in coords_by_element:
        symmetry_operation_result = apply_translations(translation_mat, element) # 12x3, 1x3
        unit_cell.append(symmetry_operation_result)
    return np.array(unit_cell)

#uses sympy package to assign x,y,z symm op from CIF file to numerical values
def apply_translations(translation_mat, xyz):
    for val, symb in zip(xyz, ['x','y','z']):
        translation_mat = translation_mat.subs(Symbol(symb), val)
    return np.array(translation_mat)
    
#extracts symm op from CIF file into a matrix of strings
def get_translation_mat(symm_operations, block):
    translation_expressions = [cif.as_string(i) for i in (block.find_loop(symm_operations))]
    translation_expressions_per_axis = Matrix([ 
        [coord_expr.strip() for coord_expr in expr.split(',')] 
        for expr in translation_expressions
    ])
    return translation_expressions_per_axis
    

#extracts fract site values and cell length and multiple for xyz coord
def get_plane(site_fract_name, cell_length_name, block):
    fract_vals = [as_number(val) for val in block.find_loop(site_fract_name)]
    cell_length = as_number(block.find_value(cell_length_name))
    return [(i * cell_length) for i in fract_vals]

# converts strings to numbers 
def as_number(num_str):
    return cif.as_number(remove_stddev(num_str))

# removes std. dev from CIF file values
def remove_stddev(num_str):
    index_of_stddev = num_str.find('(')
    if index_of_stddev >= 0:
        return num_str[:index_of_stddev]
    return num_str #string

if __name__ == '__main__':
    main()