from sympy import Symbol
import math
import numpy as np

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
    Ex, Ey, Ez = 0,0,0 # V/m
    for unit_cell in e_field_box:
        for element, charge in zip(unit_cell, charges):
            for coord in element:
                if not np.array_equal(coord, rand_coord):
                    magnitude = k * charge / (((coord[0] - rand_coord[0]) ** 2 + (coord[1] - rand_coord[1]) ** 2 + (coord[2] - rand_coord[2]) ** 2)**(3/2))
                    Ex += (coord[0] - rand_coord[0]) * magnitude
                    Ey += (coord[1] - rand_coord[1]) * magnitude
                    Ez += (coord[2] - rand_coord[2]) * magnitude
    return np.array([Ex, Ey, Ez])

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

def get_translated_cells(unit_cell, translation_permutations):
    translated_unit_cells = [] #need to generalize this to # of permutations generated from get_translation_permutations function
    for element in unit_cell: 
        translated_elements = translate_element(element, translation_permutations)
        for i, e in enumerate(translated_elements):
            if i >= len(translated_unit_cells):
                translated_unit_cells.append([])
            translated_unit_cells[i].append(e)
    return np.array(translated_unit_cells)

# gets x translations (P63mc has 12) for a single element
def translate_element(element, translation_permutations):
    translated_elements = []
    for translation in translation_permutations:
        translated_elements.append(element + np.repeat([translation], len(element), axis=0))
    return translated_elements # [directions^directions - 1, 5, 12, 3]

#creates (directions^directions - 1) directional permutations b.c excluding (0, 0, 0)
def get_translation_permutations(cell_lengths, directions):
    all_translations = []
    for x in directions:
        for y in directions:
            for z in directions:
                all_translations.append([cell_lengths[0] * x, cell_lengths[1] * y, cell_lengths[2] * z])
    return all_translations[1:]

# generates permutations e.g. 6 for given directions [1, -1], (100, 001 etc)
def get_one_choice_permutations(cell_lengths, directions):
    all_translations = []
    for dir in directions:
        for i in range(len(cell_lengths)):
            permutation = [0 for _ in cell_lengths]
            permutation[i] = dir * cell_lengths[i]
            all_translations.append(permutation)
    return all_translations

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
