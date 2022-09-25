import sys
from gemmi import cif
import numpy as np
import math
from matplotlib import pyplot as plt
from sympy import Matrix, Symbol

#for loop to calculate electric field at that random point

def main():
    doc = cif.read_file('CIF_files/BTS_Plate_300K_P63cm.cif')  # copy all the data from mmCIF file
    block = doc.sole_block()  # CIF has exactly one block
    hex_coords_by_element = get_hex_coords_by_element(block)
    hex_transformation_mat = get_hex_transformation_matrix(block)
    # cart_coords_by_element = get_cart_coords_by_element(block)
    # cart_translation_mat = get_cart_translation_matrix(block)

    print(hex_transformation_mat)
    
    unit_cell = np.array(get_unit_cell_coord(hex_transformation_mat, hex_coords_by_element))
    print("xyz=", hex_coords_by_element)
    print()
    e_field_box = np.array(get_translated_cells(unit_cell))
    e_field_box = np.append(e_field_box, np.array([unit_cell]), axis=0)
    print(e_field_box.shape)
    cart_e_field_box = get_cart_e_field_box(e_field_box)
    #print(cart_e_field_box[0])
    
    # Ba = 2+, Ti = 4+, S = 2-
    charges = np.array([[2], [4], [4], [-2], [-2]])
    charge_coords = np.repeat(charges, 12, axis=1)
    all_charge_coords = np.repeat(charge_coords[:, :, np.newaxis], 3, axis=2) # 5x12x3 array matching unit cell
    #print(all_charge_coords.shape)

    print(get_rand_coord(cart_e_field_box))
    """ UNCOMMENT TO PLOT IN CARTESIAN
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
    """

def get_rand_coord(cart_e_field_box):
    rand_coord = np.random.randint(1, cart_e_field_box.shape) # [6, 1, 2]
    return cart_e_field_box[rand_coord[0], rand_coord[1], rand_coord[2]] #cart_e_field_box[6, 1, 2]
     
""" 
(7, 5, 12, 3)
(unit cell, element, translation, xyz)
"""
# converts 7 u.c. to cartesian coord
def get_cart_e_field_box(e_field_box):
    cart_e_field_box = []
    for unit_cell in e_field_box:
        cart_unit_cell = []
        for element in unit_cell:
            cart_unit_cell.append(hex_to_cart_np(element))   
        cart_e_field_box.append(cart_unit_cell)
    cart_e_field_box = np.array(cart_e_field_box)
    return cart_e_field_box

# label the 6 translated unit cells
def get_translated_cells(unit_cell):
    a_pos, a_neg, b_pos, b_neg, c_pos, c_neg = ([] for _ in range(6))
    for element in unit_cell: 
        ap, an, bp, bn, cp, cn = translate_element(element)
        a_pos.append(ap)
        a_neg.append(an)
        b_pos.append(bp)
        b_neg.append(bn)
        c_pos.append(cp)
        c_neg.append(cn)
    return np.array(a_pos), np.array(a_neg), np.array(b_pos), np.array(b_neg), np.array(c_pos), np.array(c_neg)

#5 elements of 12x3 matrices
def translate_element(element): #12 x 3, goal: 420 xyz coords 
    a_pos = np.array([[1, 0, 0]])
    a_neg = np.array([[-1, 0, 0]])
    b_pos = np.array([[0, 1, 0]])
    b_neg = np.array([[0, -1, 0]])
    c_pos = np.array([[0, 0, 1]])
    c_neg = np.array([[0, 0, -1]])
    a_pos_cell = element + np.repeat(a_pos, 12, axis=0)
    a_neg_cell = element + np.repeat(a_neg, 12, axis=0)
    b_pos_cell = element + np.repeat(b_pos, 12, axis=0)
    b_neg_cell = element + np.repeat(b_neg, 12, axis=0)
    c_pos_cell = element + np.repeat(c_pos, 12, axis=0)
    c_neg_cell = element + np.repeat(c_neg, 12, axis=0)
    return a_pos_cell, a_neg_cell, b_pos_cell, b_neg_cell, c_pos_cell, c_neg_cell

#extract HEX atom site fract & cell length from CIF file
def get_hex_coords_by_element(block):
    hex_coords_by_element = np.array([
        get_plane('_atom_site_fract_x', '_cell_length_a', block),
        get_plane('_atom_site_fract_y', '_cell_length_b', block),
        get_plane('_atom_site_fract_z', '_cell_length_c', block)
    ]).transpose()
    # [Ba_1 Ti_1 Ti_2 S_1 S_2]
    return hex_coords_by_element

#extract HEX symmetry operations from CIF file 
def get_hex_transformation_matrix(block):
    hex_transformation_mat = get_transformation_mat('_space_group_symop_operation_xyz', block)
    return hex_transformation_mat

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
def get_unit_cell_coord(transformation_mat, coords_by_element):
    unit_cell = []
    for element in coords_by_element:
        symmetry_operation_result = apply_transformations(transformation_mat, element) # 12x3, 1x3
        unit_cell.append(symmetry_operation_result)
    return np.array(unit_cell)

#uses sympy package to assign x,y,z symm op from CIF file to numerical values
def apply_transformations(transformation_mat, xyz):
    for val, symb in zip(xyz, ['x','y','z']):
        transformation_mat = transformation_mat.subs(Symbol(symb), val)
    return np.array(transformation_mat)
    
#extracts symm op from CIF file into a matrix of strings
def get_transformation_mat(symm_operations, block):
    transformation_expressions = [cif.as_string(i) for i in (block.find_loop(symm_operations))]
    transformation_expressions_per_axis = Matrix([ 
        [coord_expr.strip() for coord_expr in expr.split(',')] 
        for expr in transformation_expressions
    ])
    return transformation_expressions_per_axis
    
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

#extract atom site fract & cell length from CIF file and convert to cartesian
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
    hex_transformation_mat = get_transformation_mat('_space_group_symop_operation_xyz', block)
    cart_transformation_mat = hex_to_cart_sympy(hex_transformation_mat)
    return cart_transformation_mat

if __name__ == '__main__':
    main()