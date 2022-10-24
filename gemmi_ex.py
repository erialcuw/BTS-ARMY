from argparse import ArgumentError
from calendar import c
import sys
from gemmi import cif
import numpy as np
import math
from matplotlib import pyplot as plt
from sympy import Matrix, Symbol

# TODO: write out 26 translations, change it to be in units of cell lengths
# can also use BC to expand the box, first expand in x, then in y, then in z, total 6 steps
def main():
    doc = cif.read_file('CIF_files/BTS_Plate_300K_P63cm.cif')  # copy all the data from mmCIF file
    block = doc.sole_block()  # CIF has exactly one block
    hex_coords_by_element = get_hex_coords_by_element(block)
    hex_transformation_mat = get_hex_transformation_matrix(block)    
    unit_cell = np.array(get_unit_cell_coord(hex_transformation_mat, hex_coords_by_element))
    print("hex_xyz=", hex_coords_by_element)
    print()
    cart_xyz = hex_to_cart_np(hex_coords_by_element)
    print("cart_xyz", cart_xyz)    

    e_field_box = np.array(get_translated_cells(unit_cell))
    print(e_field_box.shape)
    unit_cell = np.array([unit_cell])

    e_field_box = np.append(e_field_box, unit_cell, axis=0)
    print(e_field_box.shape)
    cart_e_field_box = get_cart_e_field_box(e_field_box)
    
    # Ba = 2+, Ti = 4+, S = 2-
    charges = np.array([2*1.6e-19, 4*1.6e-19, 4*1.6e-19, -2*1.6e-19, -2*1.6e-19])
    rand_coord = get_rand_coord(cart_e_field_box)
    print("rand coord =", rand_coord)
    print()
    print("E field [V/m] = ", f"{calc_e_field(cart_e_field_box, rand_coord, charges):.3e}")

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

#calculates electric field w.r.t. random coordinate value
def calc_e_field(e_field_box, rand_coord, charges):
    k = 9e9 # N * m^2/C^2
    e_total = 0 # V/m
    for unit_cell in e_field_box:
        for element, charge in zip(unit_cell, charges):
            for coord in element:
                if not np.array_equal(coord, rand_coord):
                    magnitude = k * charge / (((coord[0] - rand_coord[0]) ** 2 + (coord[1] - rand_coord[1]) ** 2 + (coord[2] - rand_coord[2]) ** 2)**(3/2))
                    e_total += math.sqrt(((magnitude * (coord[0] - rand_coord[0])) ** 2) 
                    + ((magnitude * (coord[1] - rand_coord[1])) ** 2) 
                    + ((magnitude * (coord[2] - rand_coord[2])) ** 2))
    return e_total

#gets random coordinate values
def get_rand_coord(cart_e_field_box):
    rand_index = get_rand_index(cart_e_field_box)
    return cart_e_field_box[rand_index[0], rand_index[1], rand_index[2]] #cart_e_field_box[6, 1, 2]

#gets index of random coordinate
def get_rand_index(cart_e_field_box):
    rand_index = np.random.randint(1, cart_e_field_box.shape)
    print("rand coord index = ", rand_index)  
    return rand_index    

""" 
shape of e field box: (7, 5, 12, 3)
(unit cell, element, translation, xyz)
"""
# converts 7 u.c. to cartesian coord and adds them to overall box
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
    translated_unit_cells = []
    for scalar in [1,2,3]:
        a_pos, a_neg, b_pos, b_neg, c_pos, c_neg = ([] for _ in range(6))
        for element in unit_cell: 
            ap, an, bp, bn, cp, cn = translate_element(element, scalar)
            a_pos.append(ap)
            a_neg.append(an)
            b_pos.append(bp)
            b_neg.append(bn)
            c_pos.append(cp)
            c_neg.append(cn)
        translated_unit_cells.extend([a_pos, a_neg, b_pos, b_neg, c_pos, c_neg])
    return np.array(translated_unit_cells)
"""
create unit cell for every i value and store unit cell in a rows x 3 array
do not need to worry abt saving unit cells separately
[i 0 0]
[-i 0 0]
[0 i 0]
[0 -i 0]
[0 0 i]
[0 0 -i]
"""
# creates translated unit cells composed of 5 elements of 12x3 matrices
def translate_element(element, scalar): #12 x 3, goal: 420 xyz coords 
    a_pos = np.array([[scalar, 0, 0]]) # 2 0 0
    a_neg = np.array([[-scalar, 0, 0]]) # -2 0 0
    b_pos = np.array([[0, scalar, 0]])
    b_neg = np.array([[0, -scalar, 0]])
    c_pos = np.array([[0, 0, scalar]])
    c_neg = np.array([[0, 0, -scalar]])
    a_pos_cell = element + np.repeat(a_pos, 12, axis=0)
    a_neg_cell = element + np.repeat(a_neg, 12, axis=0)
    b_pos_cell = element + np.repeat(b_pos, 12, axis=0)
    b_neg_cell = element + np.repeat(b_neg, 12, axis=0)
    c_pos_cell = element + np.repeat(c_pos, 12, axis=0)
    c_neg_cell = element + np.repeat(c_neg, 12, axis=0)
    return a_pos_cell, a_neg_cell, b_pos_cell, b_neg_cell, c_pos_cell, c_neg_cell

# extracts HEX atom site fract & cell length from CIF file
def get_hex_coords_by_element(block):
    hex_coords_by_element = np.array([
        get_plane('_atom_site_fract_x', '_cell_length_a', block),
        get_plane('_atom_site_fract_y', '_cell_length_b', block),
        get_plane('_atom_site_fract_z', '_cell_length_c', block)
    ]).transpose()
    # [Ba_1 Ti_1 Ti_2 S_1 S_2]
    return hex_coords_by_element

# extracts 12 HEX symmetry operations from CIF file 
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

# builds matrix of singular unit cell for each element's symm operation
def get_unit_cell_coord(transformation_mat, coords_by_element):
    unit_cell = []
    for element in coords_by_element:
        symmetry_operation_result = apply_transformations(transformation_mat, element) # 12x3, 1x3
        unit_cell.append(symmetry_operation_result)
    return np.array(unit_cell)

# uses sympy package to assign x,y,z symm op from CIF file to numerical values
def apply_transformations(transformation_mat, xyz):
    for val, symb in zip(xyz, ['x','y','z']):
        transformation_mat = transformation_mat.subs(Symbol(symb), val)
    return np.array(transformation_mat)
    
# extracts symmetry operations from CIF file into a matrix of strings
def get_transformation_mat(symm_operations, block):
    transformation_expressions = [cif.as_string(i) for i in (block.find_loop(symm_operations))]
    transformation_expressions_per_axis = Matrix([ 
        [coord_expr.strip() for coord_expr in expr.split(',')] 
        for expr in transformation_expressions
    ])
    return transformation_expressions_per_axis
    
# extracts fract site values and cell length and multiple for xyz coord
def get_plane(site_fract_name, cell_length_name, block):
    fract_vals = [as_number(val) for val in block.find_loop(site_fract_name)]
    cell_length = as_number(block.find_value(cell_length_name))
    return [(i * cell_length * 10e-10) for i in fract_vals]

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