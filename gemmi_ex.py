from argparse import ArgumentError
from calendar import c
import sys
from gemmi import cif
import numpy as np
import math
from matplotlib import pyplot as plt
from sympy import Matrix, Symbol

# TODO: how do I treat the r in dipole moment?
# can also use BC to expand the box, first expand in x, then in y, then in z, total 6 steps
def main():
    np.set_printoptions(precision=4)

    doc = cif.read_file('CIF_files/BTS_Plate_300K_P63cm.cif')  # copy all the data from mmCIF file
    block = doc.sole_block()  # CIF has exactly one block

    hex_coords_by_element = get_hex_coords_by_element(block) # 5 x 3
    cell_lengths = get_cell_lengths(block) # 3 x 1
    hex_transformation_mat = get_hex_transformation_matrix(block) # 12 x 3

    unit_cell = get_unit_cell_coord(hex_transformation_mat, hex_coords_by_element, cell_lengths) # 5 x 12 x 3
    print(unit_cell.shape, "Unit cell should be 5x12x3")
    translated_unit_cells = get_translated_cells(unit_cell, cell_lengths)
    print(translated_unit_cells.shape, "Translated unit cells should be (26,5,12,3) due to 26 translations")

    e_field_box = np.append([unit_cell], translated_unit_cells, axis=0)
    print(e_field_box.shape, "E field box (translations + OG UC) shape should be (27,5,12,3)")
    print("Example of Ba_1(untranslated)\n", e_field_box[0][0])
    last_translation = get_translation_permutations(cell_lengths)[-1]
    print(f"Example of Ba_1(translated({last_translation}))\n", e_field_box[-1][0])
    cart_e_field_box = get_cart_e_field_box(e_field_box)
    print("Cartesian Example of Ba_1(untranslated)\n", cart_e_field_box[0][0])
    print(f"Cartesian Example of Ba_1(translated({last_translation}))\n", cart_e_field_box[-1][0])

    # Ba = 2+, Ti = 4+, S = 2-
    charges = np.array([2*1.6e-19, 4*1.6e-19, 4*1.6e-19, -2*1.6e-19, -2*1.6e-19])
    rand_coord = get_rand_coord(cart_e_field_box)
    print("E field [V/m] = ", f"{calc_e_field(cart_e_field_box, rand_coord, charges):.4e}")
    print('Electric dipole moment [C m]', f"{calc_dipole_moment(cart_e_field_box, rand_coord, charges):.4e}")

#calculate summation of electric dipole moment [C m]
def calc_dipole_moment(e_field_box, rand_coord, charges):
    dipole_moment_total = 0
    for unit_cell in e_field_box:
        for element, charge in zip(unit_cell, charges):
            for coord in element:
                if not np.array_equal(coord, rand_coord):
                    px = charge * (coord[0] - rand_coord[0])
                    py = charge * (coord[1] - rand_coord[1])
                    pz = charge * (coord[2] - rand_coord[2])
                    dipole_moment_total += math.sqrt((px ** 2 + py ** 2 + pz ** 2))
    return dipole_moment_total

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
            cart_unit_cell.append(hex_to_cart(element))   
        cart_e_field_box.append(cart_unit_cell)
    return np.array(cart_e_field_box)

# creates array of unit cells  
def get_translated_cells(unit_cell, cell_lengths):
    translated_unit_cells = [] #need to generalize this to # of permutations generated from get_translation_permutations function
    for element in unit_cell: 
        translated_elements = translate_element(element, cell_lengths)
        for i, e in enumerate(translated_elements):
            if i >= len(translated_unit_cells):
                translated_unit_cells.append([])
            translated_unit_cells[i].append(e)
    return np.array(translated_unit_cells)

# gets x translations (P63mc has 12) for a single element
def translate_element(element, cell_lengths):
    translation_permutations = get_translation_permutations(cell_lengths)
    translated_elements = []
    for translation in translation_permutations:
        translated_elements.append(element + np.repeat([translation], 12, axis=0))
    return translated_elements # [directions^directions - 1, 5, 12, 3]

#creates 26 directional permutations excluding (0, 0, 0)
def get_translation_permutations(cell_lengths):
    directions = [0, 1, -1, 2]
    all_translations = []
    for x in directions:
        for y in directions:
            for z in directions:
                all_translations.append([cell_lengths[0] * x, cell_lengths[1] * y, cell_lengths[2] * z])
    return all_translations[1:]

# extracts HEX atom site fract & cell length from CIF file
# each row(5) represents a single element
# each column (3) represents an xyz direction
def get_hex_coords_by_element(block):
    hex_coords_by_element = np.array([
        get_fract_vals('_atom_site_fract_x', block),
        get_fract_vals('_atom_site_fract_y', block),
        get_fract_vals('_atom_site_fract_z', block)
    ]).transpose()
    # [Ba_1 Ti_1 Ti_2 S_1 S_2]
    return hex_coords_by_element

# extracts 12 HEX symmetry operations from CIF file 
def get_hex_transformation_matrix(block):
    hex_transformation_mat = get_transformation_mat('_space_group_symop_operation_xyz', block)
    return hex_transformation_mat

# converts hex to cart coordinates np.array
def hex_to_cart(hexagonal_coords):
    hex_to_cart_mat = np.array([[1, -.5, 0], [math.sqrt(3)/2, 0, 0], [0, 0, 1]])
    cartesian_coord = np.dot(hex_to_cart_mat, hexagonal_coords.T)
    return cartesian_coord.T

# builds matrix of singular unit cell for each element's symm operation
def get_unit_cell_coord(transformation_mat, coords_by_element, cell_lengths):
    unit_cell = []
    for element in coords_by_element:
        hexagonal_coords = apply_transformations(transformation_mat, element) # 12x3, 1x3
        symmetry_operation_result = multiply_by_cell_lengths(hexagonal_coords, cell_lengths)
        unit_cell.append(symmetry_operation_result)
    return np.array(unit_cell)

# uses sympy package to assign x,y,z symm op from CIF file to numerical values
def apply_transformations(transformation_mat, xyz):
    for val, symb in zip(xyz, ['x','y','z']):
        transformation_mat = transformation_mat.subs(Symbol(symb), val)
    return np.array(transformation_mat).astype(np.float128)

# Multiply each column (x,y,z) by corresponding cell_lengths (a,b,c)
def multiply_by_cell_lengths(hexagonal_coords, cell_lengths):
    for i, cell_length in enumerate(cell_lengths):
        hexagonal_coords[:, i] *= cell_length
    return hexagonal_coords
    
# extracts symmetry operations from CIF file into a matrix of strings
def get_transformation_mat(symm_operations, block):
    transformation_expressions = [cif.as_string(i) for i in (block.find_loop(symm_operations))]
    transformation_expressions_per_axis = Matrix([ 
        [coord_expr.strip() for coord_expr in expr.split(',')] 
        for expr in transformation_expressions
    ])
    return transformation_expressions_per_axis
    
# extracts fract site values
def get_fract_vals(site_fract_name, block):
    return [as_number(val) for val in block.find_loop(site_fract_name)]

# extracts all cell lengths (a,b,c)
def get_cell_lengths(block):
    cell_length_names = ['_cell_length_a', '_cell_length_b', '_cell_length_c']
    return [get_cell_length(name, block) for name in cell_length_names]

#extract single cell length given name
def get_cell_length(cell_length_name, block):
    return as_number(block.find_value(cell_length_name)) * 1e-10

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